import dill
import sys
import gc
sys.setrecursionlimit(50000)
from multiprocessing import Process, Queue
import threading
import time
import copy
import numpy as np
from model.ModelUtil import *

class SimWorker(Process):
    
    def __init__(self, input_queue, output_queue, actor, exp, model, discount_factor, action_space_continuous, 
                 settings, print_data, p, validation, eval_episode_data_queue, process_random_seed,
                 message_que, worker_id):
        super(SimWorker, self).__init__()
        self._input_queue= input_queue
        self._output_queue = output_queue
        self._eval_episode_data_queue = eval_episode_data_queue
        self._actor = actor
        self._exp = exp
        self._model = model
        self._discount_factor = discount_factor
        self._action_space_continuous= action_space_continuous
        self._settings= settings
        self._print_data=print_data
        self._p= p
        self._validation=validation
        self._max_iterations = settings['rounds'] + settings['epochs'] * 32
        self._iteration = 0    
        self._process_random_seed = process_random_seed
        ## Used to receive special messages like update your model parameters to this now!
        self._message_queue = message_que
        self._worker_id = worker_id
    
    def current_mem_usage(self):
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        except ImportError:
            return 0
        # return 0

    def setEnvironment(self, exp_):
        """
            Set the environment instance to use
        """
        self._exp = exp_
        self._model.setEnvironment(self._exp)
        
    # @profile(precision=5)
    def run(self):
        # from pympler import summary
        # from pympler import muppy
        np.random.seed(self._process_random_seed)
        import os
        ## This is no needed if there is one thread only...
        if (int(self._settings["num_available_threads"]) > 1): 
            from util.SimulationUtil import createEnvironment
            print ("Creating simulation environments for simulation workers")
            self._exp = createEnvironment(self._settings["sim_config_file"], self._settings['environment_type'], self._settings, render=self._settings['shouldRender'], index=self._worker_id)
            self._exp.setActor(self._actor)
            self._exp.getActor().init()   
            self._exp.init()
            self._exp.setRandomSeed(self._process_random_seed)
            ## The sampler might need this new model if threads > 1
            self._model.setEnvironment(self._exp)
        
        ## This get is fine, it is the first one that I want to block on.
        print ("Waiting for initial policy update.", self._message_queue)
        episodeData = self._message_queue.get()
        message = episodeData['type']
        if message == "Update_Policy":
            print ("First Message: ", message)
            data = episodeData['data']
            self._model.getPolicy().setNetworkParameters(data[5])
            if (self._settings['train_forward_dynamics']):
                self._model.getForwardDynamics().setNetworkParameters(data[6])
            self._p = data[1]
            self._model.setStateBounds(data[2])
            self._model.setActionBounds(data[3])
            self._model.setRewardBounds(data[4])
            print ("Sim worker:", os.getpid(), " State Bounds: ", self._model.getStateBounds())
            print ("Initial policy ready:")
        print ('Worker: started')
        # do some initialization here
        while True:
            eval=False
            sim_on_poli = False
            bootstrapping = False
            if (self._settings['on_policy']):
                episodeData = self._message_queue.get()
                if episodeData == None:
                    print ("Terminating worker: " , os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                    break
                elif ( episodeData['type'] == "Update_Policy" ):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Message: ", message)
                    data = episodeData['data']
                    ### Update scaling parameters
                    self._model.setStateBounds(data[2])
                    self._model.setActionBounds(data[3])
                    self._model.setRewardBounds(data[4])
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Scaling State params: ", self._model.getStateBounds())
                        print("Scaling Action params: ", self._model.getActionBounds())
                        print("Scaling Reward params: ", self._model.getRewardBounds())        
                    self._model.getPolicy().setNetworkParameters(data[5])
                    if (self._settings['train_forward_dynamics']):
                        self._model.getForwardDynamics().setNetworkParameters(data[6])
                    p = data[1]
                    if p < 0.1:
                        p = 0.1
                    self._p = p
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Sim worker:", os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                        print('\tWorker maximum memory usage: %.2f (mb)' % (self.current_mem_usage()))
                elif episodeData['type'] == "eval":
                    eval=True
                    episodeData = episodeData['data']
                    # "Sim worker evaluating episode"
                elif ( episodeData['type'] == 'sim_on_policy'):
                    sim_on_poli = True
                    episodeData = episodeData['data']
                elif ( episodeData['type'] == 'bootstrapping'):
                    bootstrapping = True
                else:
                    episodeData = episodeData['data']
                if (eval): ## No action exploration
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=0.0, validation=True, evaluation=eval)
                elif (sim_on_poli):
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                elif (bootstrapping):
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval,
                            bootstrapping=bootstrapping)
                else:    
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                self._iteration += 1
                (tuples, discounted_sum, q_value, evalData) = out
                ## Hack for now just update after ever episode
                if (eval or sim_on_poli or bootstrapping):
                    self._eval_episode_data_queue.put(out)
                else:
                    pass
            else: ## off policy, all threads sharing the same queue
                episodeData = self._input_queue.get()
                ## Check if any messages in the queue
                if episodeData == None:
                    print ("Terminating worker: " , os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                    break
                if episodeData['type'] == "eval":
                    eval=True
                    episodeData = episodeData['data']
                    # "Sim worker evaluating episode"
                elif ( episodeData['type'] == 'sim_on_policy'):
                    sim_on_poli = True
                elif ( episodeData['type'] == 'bootstrapping'):
                    bootstrapping = True
                else:
                    episodeData = episodeData['data']
                if (eval): ## No action exploration
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=0.0, validation=True, evaluation=eval)
                elif (sim_on_poli): ### With exploration
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                elif (bootstrapping): ## With exploration and noise
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval,
                            bootstrapping=bootstrapping)
                else: ##Normal??
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                self._iteration += 1
                (tuples, discounted_sum, q_value, evalData) = out
                ## Hack for now just update after ever episode
                if (eval or sim_on_poli or bootstrapping):
                    # print ("Putting episode data in queue")
                    self._eval_episode_data_queue.put(out)
                else:
                    pass
                
                ### Pull updated network parameters
                if self._message_queue.qsize() > 0:
                    data = None
                    while (not self._message_queue.empty()):
                        ## Don't block
                        try:
                            data_ = self._message_queue.get(False)
                        except Exception as inst:
                            pass
                        if (not (data_ is None)):
                            episodeData = data_
                    if (episodeData != None and (isinstance(episodeData,dict))):
                        message = episodeData['type']
                        if message == "Update_Policy":
                            data = episodeData['data']
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Message: ", message)
                            self._model.setStateBounds(data[2])
                            self._model.setActionBounds(data[3])
                            self._model.setRewardBounds(data[4])
                            self._model.getPolicy().setNetworkParameters(data[5])
                            if (self._settings['train_forward_dynamics']):
                                self._model.getForwardDynamics().setNetworkParameters(data[6])
                            p = data[1]
                            self._p = p
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Sim worker:", os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                                print('\tWorker maximum memory usage: %.2f (mb)' % (self.current_mem_usage()))

        print ("Simulation Worker Complete: ", os.getpid())
        self._exp.finish()
        return
        
    def simEpochParallel(self, actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, 
                         bootstrapping=False):
        out = simEpoch(actor, exp, model, discount_factor, anchors=anchors, action_space_continuous=action_space_continuous, settings=settings, 
                       print_data=print_data, p=p, validation=validation, epoch=epoch, evaluation=evaluation, _output_queue=self._output_queue, epsilon=settings['epsilon'],
                       bootstrapping=bootstrapping,
                       worker_id=self._worker_id)
        return out
    
    
# @profile(precision=5)
def simEpoch(actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, 
             p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=None,
             worker_id=None):
    """
        
        evaluation: If True than the simulation is being evaluated and the episodes will not terminate early.
        bootstrapping: is used to collect initial random actions for the state bounds to be calculated and to init the expBuffer
        epoch: is an integer that can be used to help create repeatable episodes to evaluation
        _output_queue: is the queue exp tuples should be put in so the learning agents can pull them out
        p:  is the probability of selecting a random action
        actor: 
    """
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
        omega = settings["omega"]
    
    action_selection = range(len(settings["discrete_actions"]))   
    reward_bounds = np.array(settings['reward_bounds'] )
    ## If tuples should be put in the output_exp_queue in batches which will include proper values for calculated future discounted rewards.
    use_batched_exp=settings['collect_tuples_in_batches']
    pa=None
    # Actor should be FIRST here
    exp.getActor().initEpoch()
    if validation:
        exp.generateValidation(anchors, epoch)
    else:
        exp.generateEnvironmentSample()
        
    exp.initEpoch()
    actor.initEpoch()
    state_ = exp.getState()
    # pa = model.predict(state_)
    if (not bootstrapping):
        q_values_ = [model.q_value(state_)]
    else:
        q_values_ = []
    viz_q_values_ = []
    discounted_sum = 0
    discounted_sums = []
    G_t = []
    G_t_rewards = []
    baseline = []
    G_ts = []
    baselines_ = []
    advantage = []
    state_num=0
    i_=0
    last_epoch_end=0
    reward_=0
    states = [] 
    actions = []
    rewards = []
    falls = []
    result_states___ = []
    exp_actions = []
    evalDatas=[]
    stds=[]
    bad_sim_state = False

    i_ = 0
    while (i_ < settings['max_epoch_length']):
        state_ = exp.getState()
        if (checkDataIsValid(state_) == True): ## Lets not wait to simulate an entire action to find out the simulation has gone haywire..
    
            if (not (visualizeEvaluation == None)):
                viz_q_values_.append(model.q_value(state_)[0])
                if (len(viz_q_values_)>30):
                     viz_q_values_.pop(0)
                visualizeEvaluation.updateLoss(viz_q_values_, np.zeros(len(viz_q_values_)))
                visualizeEvaluation.redraw()
            action=None
            if action_space_continuous:
                """
                    epsilon greedy action select
                    pa1 is best action from policy
                    ra1 is the noisy policy action action
                    ra2 is the random action
                    e is proabilty to select random action
                    0 <= e < omega < 1.0
                """
                r = np.random.rand(1)[0]
                if r < (epsilon * p) or (settings['on_policy']): # explore random actions
                    exp_action = int(1)
                    r2 = np.random.rand(1)[0]
                    if ((r2 < (omega * p))) and (not sampling) :# explore hand crafted actions
                        action = np.random.choice(action_selection)
                        action__ = actor.getActionParams(action)
                        action = action__
                    else : # add noise to current policy
                        # return ra1
                        if ( ((settings['exploration_method'] == 'gaussian_random') ) and (not sampling)):
                            pa = model.predict(state_, p=p, sim_index=worker_id, bootstrapping=bootstrapping)
                            if ( 'anneal_policy_std' in settings and (settings['anneal_policy_std'])):
                                action = randomExporation(settings["exploration_rate"] * p, pa, action_bounds)
                            else:
                                action = randomExporation(settings["exploration_rate"], pa, action_bounds)
                        elif (settings['exploration_method'] == 'gaussian_network' or 
                              (settings['use_stocastic_policy'] == True)):
                            pa_ = model.predict(state_, p=p, sim_index=worker_id, bootstrapping=bootstrapping)
                            std_ = model.predict_std(state_)
                            if ( 'anneal_policy_std' in settings and (settings['anneal_policy_std'])):
                                std_ = std_ * p
                            stds.append(std_)
                            action = randomExporationSTD(settings["exploration_rate"], pa_, std_, action_bounds)
                        elif ((settings['exploration_method'] == 'thompson')):
                            action = thompsonExploration(model, settings["exploration_rate"], state_)
                        elif ((settings['exploration_method'] == 'sampling')):
                            sim_state_ = exp.getSimState()
                            action = model.getSampler().predict(sim_state_, p=p, sim_index=worker_id, bootstrapping=bootstrapping)
                        else:
                            print ("Exploration method unknown: " + str(settings['exploration_method']))
                            sys.exit(1)
                        if (settings["use_model_based_action_optimization"] and settings["train_forward_dynamics"] ):
                            mbae_omega = settings["model_based_action_omega"]
                            if (np.random.rand(1)[0] < mbae_omega):
                                ## Need to be learning a forward dynamics deep network for this
                                if ( ('anneal_mbae' in settings) and settings['anneal_mbae'] ):
                                    mbae_lr = p * settings["action_learning_rate"]
                                else:
                                    mbae_lr = settings["action_learning_rate"]
                                if ( 'use_random_actions_for_MBAE' in settings):
                                    use_rand_act = settings['use_random_actions_for_MBAE']
                                else: 
                                    use_rand_act = False
                                (action, value_diff) = getOptimalAction(model.getForwardDynamics(), model.getPolicy(), state_, action_lr=mbae_lr, use_random_action=use_rand_act)

                else: ## exploit policy
                    exp_action = int(0) 
                    ## For sampling method to skip sampling during evaluation.
                    pa = model.predict(state_, evaluation_=evaluation, p=p, sim_index=worker_id, bootstrapping=bootstrapping)                    
                    action = pa                   
                outside_bounds=False
                action_=None
                if (settings["clamp_actions_to_stay_inside_bounds"] or (settings['penalize_actions_outside_bounds'])):
                    (action_, outside_bounds) = clampActionWarn(action, action_bounds)
                    if (settings['clamp_actions_to_stay_inside_bounds']):
                        action = action_
                if (settings["visualize_forward_dynamics"]):
                    predicted_next_state = model.getForwardDynamics().predict(np.array(state_), [action])
                    exp.visualizeNextState(predicted_next_state, action)
                    
                    action__ = model.predict(state_)
                    actions_ = []
                    dirs = []
                    deltas = np.linspace(-0.5,0.5,10)
                    for d in range(len(deltas)):
                        action_ = np.zeros_like(action__)
                        for i in range(len(action_)):
                            action_[i] = action__[i]
                        action_[0] = action__[0] + deltas[d] 
                        if ( ('anneal_mbae' in settings) and settings['anneal_mbae'] ):
                            mbae_lr = p * settings["action_learning_rate"]
                        else:
                            mbae_lr = settings["action_learning_rate"]
                        action_new_ = getOptimalAction2(model.getForwardDynamics(), model.getPolicy(), action_, state_, mbae_lr)
                        actions_.append(action_)
                        print("action_new_: ", action_new_[0], " action_: ", action_[0])
                        if ( (float(action_new_[0][0]) - float(action_[0])) > 0 ):
                            dirs.append(1.0)
                        else:
                            dirs.append(-1.0)

                    exp.getEnvironment().visualizeActions(actions_, dirs)
                    ## The perfect action?
                    exp.getEnvironment().visualizeAction(action__)
                    
                
                if (not settings["train_actor"]): # hack to use debug critic only
                    pass
                reward_ = actor.actContinuous(exp,action)
                agent_not_fell = actor.hasNotFallen(exp)
                if (outside_bounds and settings['penalize_actions_outside_bounds']):
                    reward_ = reward_ + settings['reward_lower_bound'] # TODO: this penalty should really be a function of the distance the action was outside the bounds 
            elif not action_space_continuous:
                pa = model.predict(state_)
                action = random.choice(action_selection)
                action = eGreedy(pa, action, epsilon * p)               
                action__ = actor.getActionParams(action)
                action = [action]
                reward_ = actor.actContinuous(exp, action__, bootstrapping=True)
                agent_not_fell = actor.hasNotFallen(exp)

            discounted_sum = discounted_sum + (((math.pow(discount_factor,state_num) * (reward_ )))) # *(1.0-discount_factor))
            baseline.append(model.q_value(state_)[0])            
            G_t_rewards.append(reward_)
            G_t.append(0) # *(1.0-discount_factor)))
            for i in range(len(G_t)):
                G_t[i] = G_t[i] + (((math.pow(discount_factor,(len(G_t)-i)-1) * (reward_ ))))
            resultState_ = exp.getState()

            ## For testing remove later
            if (settings["use_back_on_track_forcing"] and (not evaluation)):
                exp.getControllerBackOnTrack()
            if print_data:
                value__ = 0
                if ( not bootstrapping ):
                    value__ = model.q_value(state_)
                print ("Value: ", value__, " Action " + str(action) + " Reward: " + str(reward_) )
                if ( settings['train_reward_predictor'] and (settings['train_forward_dynamics'])):
                    predicted_reward = model.getForwardDynamics().predict_reward(state_, [action])
                    print ("Predicted reward: ", predicted_reward) 
                print ("Agent has fallen: ", not agent_not_fell )
        else:
            bad_sim_state = True
            
        ## find out if the simulation has gone haywire..
        if ( bad_sim_state or checkValidData(state_, actions, resultState_ , reward_) == False ):
            print ("Simulation is in a bad state: ")
            bad_sim_state = True
        else:
            states.extend(state_)
            actions.append(action)
            rewards.append(reward_)
            result_states___.extend(resultState_)
            if (worker_id is not None):               
                falls.append([worker_id])
            else:
                falls.append([agent_not_fell])
            exp_actions.append([exp_action])
            if (not use_batched_exp):
                if ((_output_queue != None) and (not evaluation) and (not bootstrapping)): # for multi-threading                
                    _output_queue.put((states[-1:], actions[-1:], result_states___[-1:], [rewards[-1:]],  falls[-1:], [0], exp_actions[-1:]))
            
            state_num += 1
        pa = None
        
        if ((exp.endOfEpoch() and settings['reset_on_fall'] or (bad_sim_state))):
            evalDatas.append(actor.getEvaluationData()/float(settings['max_epoch_length']))
            discounted_sums.append(discounted_sum)
            discounted_sum=0
            state_num=0
            if ('use_GAE' in settings and ( settings['use_GAE'] )):
                path = {}
                path['states'] = copy.deepcopy(states[last_epoch_end:])
                path['reward'] = np.array(rewards[last_epoch_end:])
                path["terminated"] = False
                ## Extend so that we can preserve the paths/trajectory structure.
                if (len(rewards[last_epoch_end:]) > 0):
                    advantage.extend(compute_advantage_(model, [path], discount_factor, settings['GAE_lambda']))
            else:
                if (len(rewards[last_epoch_end:]) > 0):
                    advantage.extend(discounted_rewards(np.array(rewards[last_epoch_end:]), discount_factor))
                                           
            if ( ('print_level' in settings) and (settings["print_level"]== 'debug') ):
                adv_r = [ [x, y] for x,y in zip(advantage, G_t_rewards)]
                R_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(path['reward'], G_t_rewards, G_t)]
                A_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(advantage, discounted_rewards(np.array(rewards[last_epoch_end:]), discount_factor), baseline)]
                print ("last_epoch_end: ", last_epoch_end, " i_ ", i_)
                print("Advantage, R: ", adv_r)
                print ("Lengths: ", len(rewards[last_epoch_end:]), len(G_t_rewards), len(G_t))
                print ("Rewards: ", R_r)
                print ("Advantage, discounted Reward, baseline: ", np.array(A_r))
                
            G_ts.extend(copy.deepcopy(G_t))
            baselines_.extend(copy.deepcopy(baseline))
            if (use_batched_exp):
                if ((_output_queue != None) and (not evaluation) and (not bootstrapping)): # for multi-threading
                    tmp_states = copy.deepcopy(states [last_epoch_end:])
                    tmp_actions = copy.deepcopy(actions[last_epoch_end:])
                    tmp_rewards = copy.deepcopy(rewards[last_epoch_end:])
                    tmp_falls = copy.deepcopy(falls[last_epoch_end:])
                    tmp_result_states = copy.deepcopy(result_states___[last_epoch_end:])
                    tmp_G_ts = copy.deepcopy(G_ts[last_epoch_end:])
                    tmp_exp_actions = copy.deepcopy(exp_actions[last_epoch_end:])
                    
                    for state__, action__, reward__, result_state__, fall__, G_t__, exp_actions__ in zip(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls, tmp_G_ts, tmp_exp_actions):
                        _output_queue.put((state__, action__, result_state__, reward__, fall__, G_t__, exp_actions__))
            
            last_epoch_end=i_+1
            
            G_t = []
            G_t_rewards = []
            baseline = []
            exp.getActor().initEpoch()
            if validation:
                exp.generateValidation(anchors, (epoch * settings['max_epoch_length']) + i_)
            else:
                exp.generateEnvironmentSample()
                
            exp.initEpoch()
            state_ = exp.getState()
            if (not bootstrapping):
                q_values_.append(model.q_value(state_))
                
        if ( bad_sim_state ):
            print ("bad state, fixing i_: ")
        else:
            i_ += 1
        bad_sim_state = False
        
    
    evalDatas.append(actor.getEvaluationData()/float(settings['max_epoch_length']))
    evalData = [np.mean(evalDatas)]
    discounted_sums.append(discounted_sum)
    G_ts.extend(copy.deepcopy(G_t))
    baselines_.extend(copy.deepcopy(baseline))
    discounted_sum = G_ts
    q_value = baselines_
    
    if print_data:
        print ("Evaluation: ", str(evalData))
        print ("Eval Datas: ", evalDatas) 

    if (use_batched_exp):
        if ((_output_queue != None) and (not evaluation) and (not bootstrapping)): # for multi-threading
            tmp_states = copy.deepcopy(states[last_epoch_end:])
            tmp_actions = copy.deepcopy(actions[last_epoch_end:])
            tmp_rewards = copy.deepcopy(rewards[last_epoch_end:])
            tmp_falls = copy.deepcopy(falls[last_epoch_end:])
            tmp_result_states = copy.deepcopy(result_states___[last_epoch_end:])
            tmp_G_ts = copy.deepcopy(G_ts[last_epoch_end:])
            tmp_exp_actions = copy.deepcopy(exp_actions[last_epoch_end:])
            
            for state__, action__, reward__, result_state__, fall__, G_t__, exp_actions__ in zip(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls, tmp_G_ts, tmp_exp_actions):
                _output_queue.put((state__, action__, result_state__, reward__, fall__, G_t__, exp_actions__))
    ## Compute Advantage
 
    if ('use_GAE' in settings and ( settings['use_GAE'] )):
        path = {}
        path['states'] = copy.deepcopy(states [last_epoch_end:])
        path['reward'] = np.array(rewards[last_epoch_end:])
        path["terminated"] = False
        ## Extend so that we can preserve the paths/trajectory structure.
        if (len(rewards[last_epoch_end:]) > 0):
            advantage.extend(compute_advantage_(model, [path], discount_factor, settings['GAE_lambda']))
    else:
        if (len(rewards[last_epoch_end:]) > 0):
            advantage.extend(discounted_rewards(np.array(rewards[last_epoch_end:]), discount_factor))
        

    if ( ('print_level' in settings) and (settings["print_level"]== 'debug') ):
        adv_r = [ [x, y] for x,y in zip(advantage, G_t_rewards)]
        R_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(path['reward'], G_t_rewards, G_t)]
        A_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(advantage, discounted_rewards(np.array(rewards[last_epoch_end:]), discount_factor), baseline)]
        print ("last_epoch_end: ", last_epoch_end, " i_ ", i_)
        print("Advantage, R: ", adv_r)
        print ("Lengths: ", len(rewards[last_epoch_end:]), len(G_t_rewards), len(G_t))
        print ("Rewards: ", R_r)
        print ("Advantage, discounted Reward, baseline: ", np.array(A_r))
    advantage = np.reshape(np.array([advantage]), newshape=(-1,1))
    tuples = (states, actions, result_states___, rewards, falls, G_ts, advantage, exp_actions)

    return (tuples, discounted_sum, q_value, evalData)
    

# @profile(precision=5)
def evalModel(actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, 
              settings=None, print_data=False, p=0.0, evaluation=False, visualizeEvaluation=None,
              bootstrapping=False, sampling=False):
    print ("Evaluating model:")
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    for i in range(anchors): # half the anchors
        (tuples, discounted_sum, value, evalData) = simEpoch(actor, exp, 
                model, discount_factor, anchors=i, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=print_data, p=p, validation=True, epoch=epoch_, evaluation=evaluation,
                visualizeEvaluation=visualizeEvaluation, bootstrapping=bootstrapping, sampling=sampling, epsilon=settings['epsilon'])
        epoch_ = epoch_ + 1
        (states, actions, result_states, rewards, falls, G_t, advantage, exp_actions) = tuples
        if model.getExperience().samples() >= settings['batch_size']:
            _states, _actions, _result_states, _rewards, falls, _G_ts, exp_actions = model.getExperience().get_batch(settings['batch_size'])
            error = model.bellman_error(_states, _actions, _rewards, _result_states, falls)
        else :
            error = [[0]]
            print ("Error: not enough samples in experience to check bellman error: ", model.getExperience().samples(), " needed " , settings['batch_size'] )
        error = np.mean(np.fabs(error))
        discounted_values.append(discounted_sum)
        values.append(value)
        # This works better because epochs can terminate early, which is bad.
        reward_over_epocs.append(np.mean(np.array(rewards)))
        bellman_errors.append(error)
        evalDatas.append(evalData)
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        print ("Reward for best epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
        print ("reward_over_epocs" + str(reward_over_epocs))
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['debug']):
        print ("Discounted sum: ", discounted_values)
        print ("Initial values: ", values)
    mean_reward = np.mean(reward_over_epocs)
    std_reward = np.std(reward_over_epocs)
    mean_bellman_error = np.mean(bellman_errors)
    std_bellman_error = np.std(bellman_errors)
    mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
    std_discount_error = np.std(np.array(discounted_values) - np.array(values))
    mean_eval = np.mean(evalDatas)
    std_eval = np.std(evalDatas)
    
    discounted_values = []
    reward_over_epocs = []
    bellman_errors = []
        
    return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)

# @profile(precision=5)
def evalModelParrallel(input_anchor_queue, eval_episode_data_queue, model, settings, anchors=None):
    print ("Evaluating model Parrallel:")
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    i = 0 
    while i < anchors: # half the anchors       
        j = 0
        while (j < settings['num_available_threads']) and ( (i + j) < anchors):
            episodeData = {}
            episodeData['data'] = i
            episodeData['type'] = 'eval'
            if (settings['on_policy']):
                input_anchor_queue[j].put(episodeData)
            else:
                input_anchor_queue.put(episodeData)
            j += 1
            
        j = 0
        while (j < settings['num_available_threads']) and ( (i + j) < anchors):
            (tuples, discounted_sum, value, evalData) =  eval_episode_data_queue.get()
            j += 1
            epoch_ = epoch_ + 1
            (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions) = tuples
            if model.getExperience().samples() >= settings['batch_size']:
                _states, _actions, _result_states, _rewards, falls, _G_ts, exp_actions = model.getExperience().get_batch(settings['batch_size'])
                error = model.bellman_error(_states, _actions, _rewards, _result_states, falls)                
            else :
                error = [[0]]
                print ("Error: not enough samples in experience to check bellman error: ", model.getExperience().samples(), " needed " , settings['batch_size'])
            error = np.mean(np.fabs(error))
            discounted_values.append(discounted_sum)
            values.append(value)
            reward_over_epocs.append(np.mean(np.array(rewards)))
            bellman_errors.append(error)
            evalDatas.append(evalData)
        i += j
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        print ("Reward for best epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
        print ("reward_over_epocs" + str(reward_over_epocs))
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['debug']):
        print ("Discounted sum: ", np.array(discounted_values))
        print ("Initial values: ", np.array(values))
        for i in range(len(discounted_values)):
            print ("len(discounted_values[i]): ", len(discounted_values[i]), " len(discounted_values[i]): ", 
                   len(values[i]))
    mean_reward = np.mean(reward_over_epocs)
    std_reward = np.std(reward_over_epocs)
    mean_bellman_error = np.mean(bellman_errors)
    std_bellman_error = np.std(bellman_errors)
    mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
    std_discount_error = np.std(np.array(discounted_values) - np.array(values))
    mean_eval = np.mean(evalDatas)
    std_eval = np.std(evalDatas)
    
    discounted_values = []
    reward_over_epocs = []
    bellman_errors = []
        
    return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)

# @profile(precision=5)
def simModelParrallel(sw_message_queues, eval_episode_data_queue, model, settings, anchors=None, type=None):
    print ("Simulating epochs in Parallel:")
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    states = []
    actions = []
    result_states = []
    rewards = []
    falls = []
    G_ts = []
    advantage = [] 
    exp_actions = []
    discounted_sum = []
    value = []
    evalData = []
    i = 0 
    while i < anchors: # half the anchors        
        j = 0
        while (j < settings['num_available_threads']) and ( (i + j) < anchors):
            episodeData = {}
            episodeData['data'] = i
            if ( (type is None) ):
                episodeData['type'] = 'sim_on_policy'
            else:
                episodeData['type'] = 'bootstrapping'
            # sw_message_queues[j].put(episodeData)
            if (settings['on_policy']):
                sw_message_queues[j].put(episodeData)
            else:
                sw_message_queues.put(episodeData)
            j += 1
            
        # for anchs in anchors: # half the anchors
        j = 0
        while (j < settings['num_available_threads']) and ( (i + j) < anchors):
            (tuples, discounted_sum_, value_, evalData_) =  eval_episode_data_queue.get()
            discounted_sum.append(discounted_sum_)
            value.append(value_)
            evalData.append(evalData_)
            j += 1
            epoch_ = epoch_ + 1
            (states_, actions_, result_states_, rewards_, falls_, G_ts_, advantage_, exp_actions_) = tuples
            states.extend(states_)
            actions.extend(actions_)
            result_states.extend(result_states_)
            rewards.extend(rewards_)
            falls.extend(falls_)
            G_ts.extend(G_ts_)
            advantage.extend(advantage_)
            exp_actions.extend(exp_actions_)
        i += j
        
    tuples = (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions)
    return (tuples, discounted_sum, value, evalData)
        
# @profile(precision=5)
def collectExperience(actor, exp_val, model, settings, sim_work_queues=None, 
                      eval_episode_data_queue=None):
    from util.ExperienceMemory import ExperienceMemory
    
    ## Easy hack to fix issue with training for MBAE needing a LearningAgent with forward dyanmics model and not just algorithm
    settings = copy.deepcopy(settings)
    settings['use_model_based_action_optimization'] = False
    action_selection = range(len(settings["discrete_actions"]))
    print ("Action selection: " + str(action_selection))
    reward_bounds=np.array(settings["reward_bounds"])
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    state_bounds = np.array(settings['state_bounds'], dtype=float)
    
    if (settings["bootsrap_with_discrete_policy"]) and (settings['bootstrap_samples'] > 0):
        (states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions) = collectExperienceActionsContinuous(actor, exp_val, model, settings['bootstrap_samples'], settings=settings, action_selection=action_selection, sim_work_queues=sim_work_queues, 
                                                                                                                   eval_episode_data_queue=eval_episode_data_queue)
        
        print (" Shape Actions: ", actions.shape)
        print (" Shape result states: ", resultStates.shape)
        print (" Shape rewards_: ", rewards_.shape)
        print (" Shape falls: ", falls_.shape)
        print (" Shape advantage: ", G_ts_.shape)
        print (" Shape exp_actions: ", exp_actions.shape)
        
        state_bounds = np.ones((2,states.shape[1]))
        
        state_avg = states[:settings['bootstrap_samples']].mean(0)
        state_stddev = states[:settings['bootstrap_samples']].std(0)
        reward_avg = rewards_[:settings['bootstrap_samples']].mean(0)
        reward_stddev = rewards_[:settings['bootstrap_samples']].std(0)
        action_avg = actions[:settings['bootstrap_samples']].mean(0)
        action_stddev = actions[:settings['bootstrap_samples']].std(0)
        print("Computed state min bound: ", state_avg - state_stddev)
        print("Computed state max bound: ", state_avg + state_stddev)
        if (settings['state_normalization'] == "minmax"):
            state_bounds[0] = states[:settings['bootstrap_samples']].min(0)
            state_bounds[1] = states[:settings['bootstrap_samples']].max(0)
        elif (settings['state_normalization'] == "variance"):
            state_bounds[0] = state_avg - (state_stddev * 2.0)
            state_bounds[1] = state_avg + (state_stddev * 2.0)
            reward_bounds[0] = reward_avg - (reward_stddev)
            reward_bounds[1] = reward_avg + (reward_stddev)
        elif (settings['state_normalization'] == "given"):
            # pass # Use bound specified in file
            state_bounds = np.array(settings['state_bounds'], dtype=float)
        else:
            print ("State scaling strategy unknown: ", (settings['state_normalization']))
            
        ## Cast data to the proper type
        state_bounds = np.array(state_bounds, dtype=settings['float_type'])
        reward_bounds = np.array(reward_bounds, dtype=settings['float_type'])
        action_bounds = np.array(action_bounds, dtype=settings['float_type'])
            
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings = settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
        experience.setSettings(settings)
        
        
        print ("State Mean:" + str(state_avg))
        print ("State Variance: " + str(state_stddev))
        print ("Reward Mean:" + str(reward_avg))
        print ("Reward Variance: " + str(reward_stddev))
        print ("Action Mean:" + str(action_avg))
        print ("Action Variance: " + str(action_stddev))
        print ("Max State:" + str(state_bounds[1]))
        print ("Min State:" + str(state_bounds[0]))
        print ("Max Reward:" + str(reward_bounds[1]))
        print ("Min Reward:" + str(reward_bounds[0]))
        print ("Max Action:" + str(action_bounds[1]))
        print ("Min Action:" + str(action_bounds[0]))
        
        experience.setStateBounds(state_bounds)
        experience.setRewardBounds(reward_bounds)
        experience.setActionBounds(action_bounds)
        
        for state, action, resultState, reward_, fall_, G_t, exp_action in zip(states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions):            
            if settings['action_space_continuous']:                
                experience.insertTuple(([state], [action], [resultState], [reward_], [fall_], G_t, [exp_action]))
            else:
                experience.insertTuple(([state], [action], [resultState], [reward_], [falls_], G_t, [exp_action]))
    else: ## Most likely performing continuation learning
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(model.getStateBounds()[0]), len(model.getActionBounds()[0]), settings['expereince_length'], continuous_actions=True, settings = settings)
        else:
            experience = ExperienceMemory(len(model.getStateBounds()[0]), 1, settings['expereince_length'])
            experience.setSettings(settings)
            experience.setStateBounds(model.getStateBounds())
            experience.setRewardBounds(model.getRewardBounds())
            experience.setActionBounds(model.getActionBounds())
    return  experience, state_bounds, reward_bounds, action_bounds

# @profile(precision=5)
def collectExperienceActionsContinuous(actor, exp, model, samples, settings, action_selection, sim_work_queues=None, eval_episode_data_queue=None):
    i = 0
    states = []
    actions = []
    resultStates = []
    rewards = []
    falls = []
    G_ts = []
    exp_actions = []
    episode_ = 0
    while i < samples:
        ## Actor should be FIRST here
        if ( ( sim_work_queues is None ) or (eval_episode_data_queue is None)):
            out = simEpoch(actor=actor, exp=exp, model=model, discount_factor=settings['discount_factor'], anchors=episode_, 
                               action_space_continuous=settings['action_space_continuous'], settings=settings, print_data=False,
                                p=1.0, validation=settings['train_on_validation_set'], bootstrapping=True, epsilon=1.0)
        else:
            out = simModelParrallel( sw_message_queues=sim_work_queues,
                                 model=model, settings=settings, 
                                 eval_episode_data_queue=eval_episode_data_queue, 
                                 anchors=settings['epochs'],
                                 type='bootstrapping')
        (tuples, discounted_sum_, q_value_, evalData) = out
        (states_, actions_, result_states_, rewards_, falls_, G_t_, advantage, exp_actions_) = tuples
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print ("Shape other states_: ", np.array(states_).shape)
            print ("Shape other action_: ", np.array(actions_).shape)
        states.extend(states_)
        actions.extend(actions_)
        rewards.extend(rewards_)
        resultStates.extend(result_states_)
        falls.extend(falls_)
        G_ts.extend(G_t_)
        exp_actions.extend(exp_actions_)
        
        i=i+len(states_)
        episode_ += 1
        episode_ = episode_ % settings["epochs"]
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print("Number of Experience samples so far: ", i)
        

    print ("Done collecting experience.")
    return (np.array(states), np.array(actions), np.array(resultStates), np.array(rewards), 
            np.array(falls), np.array(G_ts), np.array(exp_actions))  


def modelEvaluationParallel(settings_file_name):
    
    from model.ModelUtil import getSettings
    import multiprocessing
    
    settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    

    from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience
    from model.LearningAgent import LearningAgent, LearningWorker
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize

    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    batch_size=settings["batch_size"]
    state_bounds = np.array(settings['state_bounds'])
    action_space_continuous=settings["action_space_continuous"]  
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0]
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
        
    input_anchor_queue = multiprocessing.Queue(settings['queue_size_limit'])
    output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
    eval_episode_data_queue = multiprocessing.Queue(settings['num_available_threads'])
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    namespace.p=0
    
    exp_val = None
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    
    ### Using a wrapper for the type of actor now
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    # actor = ActorInterface(discrete_actions)
    actor = createActor(str(settings['environment_type']),settings, experience)
    masterAgent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    
    file_name=directory+getAgentName()+"_Best.pkl"
    f = open(file_name, 'r')
    model = dill.load(f)
    f.close()
    print ("State Length: ", len(model.getStateBounds()[0]) )
    
    if (settings['train_forward_dynamics']):
        file_name_dynamics=directory+"forward_dynamics_"+"_Best.pkl"
        # file_name=directory+getAgentName()+".pkl"
        f = open(file_name_dynamics, 'r')
        forwardDynamicsModel = dill.load(f)
        f.close()
    
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'r')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    # this is the process that selects which game to play
    
    sim_workers = []
    for process in range(settings['num_available_threads']):
        # this is the process that selects which game to play
        exp_=None
        
        if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
            print ("Assigning same EXP")
            exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
        print ("original exp: ", exp_)
        if ( settings['use_simulation_sampling'] ):
            
            sampler = createSampler(settings, exp_)
            ## This should be some kind of copy of the simulator not a network
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp_)
            sampler.setForwardDynamics(forwardDynamicsModel)
            # sampler.setPolicy(model)
            agent = sampler
            print ("thread together exp: ", agent._exp)
            # sys.exit()
        else:
            agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        
        agent.setSettings(settings)
        w = SimWorker(namespace, input_anchor_queue, output_experience_queue, actor, exp_, agent, discount_factor, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=False, p=0.0, validation=True, eval_episode_data_queue=eval_episode_data_queue, process_random_seed=settings['random_seed']+process )
        sim_workers.append(w)
        
    if (int(settings["num_available_threads"]) != 1): # This is okay if there is one thread only...
            for sw in sim_workers:
                print ("Sim worker")
                print (sw)
                sw.start()
            
    ## This needs to be done after the simulation work processes are created
    exp_val = createEnvironment(str(settings["forwardDynamics_config_file"]), settings['environment_type'], settings, render=settings['shouldRender'])
    exp_val.setActor(actor)
    exp_val.getActor().init()
    exp_val.init()
    
    exp = exp_val
    exp.setActor(actor)
    if (settings['train_forward_dynamics']):
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
    
    exp.getActor().init()   
    exp.init()
    if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
        sim_workers[0].setEnvironment(exp_val)
        sim_workers[0].start()
        
    expected_value_viz=None
    if (settings['visualize_expected_value']):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)

    namespace.agentPoly = masterAgent.getPolicy().getNetworkParameters()
    namespace.model = model
        
    for k in range(5):
        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModelParrallel( input_anchor_queue=input_anchor_queue,
                                                                model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
    
        print ("Mean eval: ", mean_eval)
    print ("Mean Evaluation: " + str(mean_eval))
    
    print ("Terminating Workers")
    for sw in sim_workers: # Should update these more offten
        input_anchor_queue.put(None)
        
    for sw in sim_workers: # Should update these more offten
        sw.join()
        

def modelEvaluation(settings_file_name, runLastModel=False, settings=None):
    
    from model.ModelUtil import getSettings
    if (settings is None):
        settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    ## Theano needs to be imported after the flags are set.
    # from ModelEvaluation import *
    # from model.ModelUtil import *
    from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience
    # from model.ModelUtil import validBounds
    from model.LearningAgent import LearningAgent, LearningWorker
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler, getAgentName
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    # from model.ModelUtil import *
    # from actor.ActorInterface import *
    # from util.SimulationUtil import *

    settings['shouldRender'] = True 
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    batch_size=settings["batch_size"]
    state_bounds = np.array(settings['state_bounds'])
    action_space_continuous=settings["action_space_continuous"]  
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0]
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    
    ### Using a wrapper for the type of actor now
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    actor = createActor(str(settings['environment_type']),settings, experience)
    
    sim_index=0
    if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
        sim_index = settings['override_sim_env_id']
    exp = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=True, index=sim_index)
    
    if ( settings['use_simulation_sampling'] ):
        sampler = createSampler(settings, exp)
        ## This should be some kind of copy of the simulator not a network
        forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp)
        sampler.setForwardDynamics(forwardDynamicsModel)
        masterAgent = sampler
        print ("thread together exp: ", masterAgent._exp)
    else:
        masterAgent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                                  action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    
    if (runLastModel == True):
        file_name=directory+getAgentName()+".pkl"
    else:
        file_name=directory+getAgentName()+"_Best.pkl"
    print("Loading model: ", file_name)
    f = open(file_name, 'rb')
    model = dill.load(f)
    f.close()
    print ("State Length: ", len(model.getStateBounds()[0]) )
    
    if (settings['train_forward_dynamics']):
        if (runLastModel == True):
            file_name_dynamics=directory+"forward_dynamics_"+".pkl"
        else:
            file_name_dynamics=directory+"forward_dynamics_"+"_Best.pkl"
        f = open(file_name_dynamics, 'rb')
        forwardDynamicsModel = dill.load(f)
        f.close()
    
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    if (settings['train_forward_dynamics']):
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)

    
    exp.setActor(actor)
    exp.getActor().init()   
    exp.init()
    expected_value_viz=None
    if (settings['visualize_expected_value']):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, masterAgent, discount_factor, anchors=settings['eval_epochs'], 
                                                                                                                        action_space_continuous=action_space_continuous, settings=settings, print_data=True, evaluation=True,
                                                                                                                        visualizeEvaluation=expected_value_viz)
    print ("Average Reward: " + str(mean_reward))
    
    
if __name__ == "__main__":
    """
        If a third param is specified run in the last saved model not the best model.
    """
    import time
    import datetime
    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
    print("options: ", options)
    print("options['configFile']: ", options['configFile'])
        
    
    
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()
    
    for option in options:
        if ( not (options[option] is None) ):
            print ("Updateing option: ", option, " = ", options[option])
            settings[option] = options[option]
        
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    
    modelEvaluation(sys.argv[1], runLastModel=True, settings=settings)
    
