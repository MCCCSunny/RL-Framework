import copy
import sys
sys.setrecursionlimit(50000)
import os
import json
sys.path.append("../")
sys.path.append("../characterSimAdapter/")
import math
import numpy as np

import random
import dill
import dill as pickle
import dill as cPickle

import cProfile, pstats, io
import gc

import multiprocessing


sim_processes = []
learning_processes = []
_input_anchor_queue = None
_output_experience_queue = None
_eval_episode_data_queue = None
_sim_work_queues = []

def createLearningAgent(settings, output_experience_queue, state_bounds, action_bounds, reward_bounds):
    """
        Create the Learning Agent to be used
    """
    from model.LearningAgent import LearningAgent, LearningWorker
    
    learning_workers = []
    for process in range(1):
        agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        
        agent.setSettings(settings)
        
        lw = LearningWorker(output_experience_queue, agent, random_seed_=settings['random_seed']+process + 1)
        learning_workers.append(lw)  
    masterAgent = agent
    return (agent, learning_workers)

def createSimWorkers(settings, input_anchor_queue, output_experience_queue, eval_episode_data_queue, model, forwardDynamicsModel, exp_val, state_bounds, action_bounds, reward_bounds, default_sim_id=None):
    """
        Creates a number of simulation workers and the message queues that
        are used to tell them what to simulate.
    """
    
    from model.LearningAgent import LearningAgent, LearningWorker
    from ModelEvaluation import SimWorker
    from util.SimulationUtil import createActor, getAgentName
    
    sim_workers = []
    sim_work_queues = []
    for process in range(settings['num_available_threads']):
        # this is the process that selects which game to play
        exp_=None
        
        if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
            print ("Assigning same EXP")
            exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
        print ("original exp: ", exp_)
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None) #actor environment
        
        agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        
        agent.setSettings(settings)
        agent.setPolicy(model)
        if (settings['train_forward_dynamics']):
            agent.setForwardDynamics(forwardDynamicsModel)
        
        elif (settings['use_simulation_sampling']):
            
            sampler = createSampler(settings, exp_)
            ## This should be some kind of copy of the simulator not a network
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp_)
            sampler.setForwardDynamics(forwardDynamicsModel)
            # sampler.setPolicy(model)
            agent.setSampler(sampler)
            print ("thread together exp: ", sampler._exp)
        
        ### Check if this is to be a mult-task simulation
        if type(settings['sim_config_file']) is list:
            if (default_sim_id != None):
                sim_id = default_sim_id
            else:
                sim_id = process
        else:
            sim_id = 0
            
        if (settings['on_policy']):
            message_queue = multiprocessing.Queue(1)
        else:
            message_queue = multiprocessing.Queue(settings['epochs'])
        sim_work_queues.append(message_queue)
        w = SimWorker(input_anchor_queue, output_experience_queue, actor, exp_, agent, settings["discount_factor"], action_space_continuous=settings['action_space_continuous'], 
                settings=settings, print_data=False, p=0.0, validation=True, eval_episode_data_queue=eval_episode_data_queue, process_random_seed=settings['random_seed']+process + 1,
                message_que=message_queue, worker_id=sim_id )
        sim_workers.append(w)
    
    return (sim_workers, sim_work_queues)
    

def trainModelParallel(inputData):
        settingsFileName = inputData[0]
        settings = inputData[1]
        np.random.seed(int(settings['random_seed']))
        import os    
        if ( 'THEANO_FLAGS' in os.environ): 
            os.environ['THEANO_FLAGS'] = os.environ['THEANO_FLAGS']+"mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
        else:
            os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
        import keras.backend
        keras.backend.set_floatx(settings['float_type'])
        print ("K.floatx()", keras.backend.floatx())
        
        from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience, simEpoch, evalModel, simModelParrallel
        from model.ModelUtil import validBounds, fixBounds, anneal_value
        from model.LearningAgent import LearningAgent, LearningWorker
        from util.SimulationUtil import validateSettings
        from util.SimulationUtil import createEnvironment
        from util.SimulationUtil import createRLAgent
        from util.SimulationUtil import createActor, getAgentName
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
        
        
        from util.ExperienceMemory import ExperienceMemory
        from RLVisualize import RLVisualize
        from NNVisualize import NNVisualize
        
        #from sim.PendulumEnvState import PendulumEnvState
        #from sim.PendulumEnv import PendulumEnv
        #from sim.BallGame2DEnv import BallGame2DEnv
        settings = validateSettings(settings)
        
        model_type= settings["model_type"]
        directory= getDataDirectory(settings)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
                       
        # copy settings file
        out_file_name=directory+os.path.basename(settingsFileName)
        print ("Saving settings file with data: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(settings, indent=4))
        out_file.close()
        ### Try and save algorithm and model files for reference
        if "." in settings['model_type']:
            ### convert . to / and copy file over
            file_name = settings['model_type']
            k = file_name.rfind(".")
            file_name = file_name[:k]
            file_name_read = file_name.replace(".", "/")
            file_name_read = file_name_read + ".py"
            print ("model file name:", file_name)
            print ("os.path.basename(file_name): ", os.path.basename(file_name))
            file = open(file_name_read, 'r')
            out_file = open(directory+file_name+".py", 'w')
            out_file.write(file.read())
            file.close()
            out_file.close()
        if "." in settings['agent_name']:
            ### convert . to / and copy file over
            file_name = settings['agent_name']
            k = file_name.rfind(".")
            file_name = file_name[:k]
            file_name_read = file_name.replace(".", "/")
            file_name_read = file_name_read + ".py"
            print ("model file name:", file_name)
            print ("os.path.basename(file_name): ", os.path.basename(file_name))
            file = open(file_name_read, 'r')
            out_file = open(directory+file_name+".py", 'w')
            out_file.write(file.read())
            file.close()
            out_file.close()
            
        if (settings['train_forward_dynamics']):
            if "." in settings['forward_dynamics_model_type']:
                ### convert . to / and copy file over
                file_name = settings['forward_dynamics_model_type']
                k = file_name.rfind(".")
                file_name = file_name[:k]
                file_name_read = file_name.replace(".", "/")
                file_name_read = file_name_read + ".py"
                print ("model file name:", file_name)
                print ("os.path.basename(file_name): ", os.path.basename(file_name))
                file = open(file_name_read, 'r')
                out_file = open(directory+file_name+".py", 'w')
                out_file.write(file.read())
                file.close()
                out_file.close()
            
        rounds = settings["rounds"]
        epochs = settings["epochs"]
        epsilon = settings["epsilon"]
        discount_factor=settings["discount_factor"]
        reward_bounds=np.array(settings["reward_bounds"])
        batch_size=settings["batch_size"]
        train_on_validation_set=settings["train_on_validation_set"]
        state_bounds = np.array(settings['state_bounds'])
        discrete_actions = np.array(settings['discrete_actions']) #9*6
        num_actions= discrete_actions.shape[0] # number of rows
        print ("Sim config file name: " + str(settings["sim_config_file"]))
        action_space_continuous=settings['action_space_continuous']

        if (settings['num_available_threads'] == 1):
            input_anchor_queue = multiprocessing.Queue(settings['queue_size_limit'])
            input_anchor_queue_eval = multiprocessing.Queue(settings['queue_size_limit'])
            output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
            eval_episode_data_queue = multiprocessing.Queue(settings['queue_size_limit'])
        else:
            input_anchor_queue = multiprocessing.Queue(settings['epochs'])
            input_anchor_queue_eval = multiprocessing.Queue(settings['epochs'])
            output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
            eval_episode_data_queue = multiprocessing.Queue(settings['eval_epochs'])
            
        if (settings['on_policy']): ## So that off policy agent does not learn
            output_experience_queue = None
            
        sim_work_queues = []
        
        action_space_continuous=settings['action_space_continuous']
        if action_space_continuous:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None)
        exp_val = None
        if (not validBounds(action_bounds)):
            # Check that the action bounds are spcified correctly
            print("Action bounds invalid: ", action_bounds)
            sys.exit()
        if (not validBounds(state_bounds)):
            # Probably did not collect enough bootstrapping samples to get good state bounds.
            print("State bounds invalid: ", state_bounds)
            state_bounds = fixBounds(np.array(state_bounds))
            bound_fixed = validBounds(state_bounds)
            print("State bounds fixed: ", bound_fixed)
            sys.exit()
        if (not validBounds(reward_bounds)):
            print("Reward bounds invalid: ", reward_bounds)
            sys.exit()
        
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
            
        experience.setSettings(settings)
        
        if settings['visualize_learning']:
            title = settings['agent_name']
            k = title.rfind(".") + 1
            if (k > len(title)): ## name does not contain a .
                k = 0 
            title = title[k:]    
            rlv = RLVisualize(title=title + " agent on " + str(settings['environment_type']), settings=settings)
            rlv.setInteractive()
            rlv.init()
        if (settings['train_forward_dynamics']):
            if settings['visualize_learning']:
                title = settings['forward_dynamics_model_type']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                nlv = NNVisualize(title=str("Dynamics Model") + " with " + title, settings=settings)
                nlv.setInteractive()
                nlv.init()
        if (settings['train_reward_predictor']):
            if settings['visualize_learning']:
                title = settings['forward_dynamics_model_type']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                
                title = title[k:]
                rewardlv = NNVisualize(title=str("Reward Model") + " with " + title, settings=settings)
                rewardlv.setInteractive()
                rewardlv.init()
                 
        if (settings['debug_critic']): #True
            criticLosses = []
            criticRegularizationCosts = [] 
            if (settings['visualize_learning']):
                title = settings['agent_name']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + title)
                critic_loss_viz.setInteractive()
                critic_loss_viz.init()
                critic_regularization_viz = NNVisualize(title=str("Critic Reg Cost") + " with " + title)
                critic_regularization_viz.setInteractive()
                critic_regularization_viz.init()
            
        if (settings['debug_actor']): # True
            actorLosses = []
            actorRegularizationCosts = []            
            if (settings['visualize_learning']): #False
                title = settings['agent_name']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                actor_loss_viz = NNVisualize(title=str("Actor Loss") + " with " + title)
                actor_loss_viz.setInteractive()
                actor_loss_viz.init()
                actor_regularization_viz = NNVisualize(title=str("Actor Reg Cost") + " with " + title)
                actor_regularization_viz.setInteractive()
                actor_regularization_viz.init()
        
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings) #return a model class
        forwardDynamicsModel = None
        if (settings['train_forward_dynamics']): #False
            if (settings['forward_dynamics_model_type'] == "SingleNet"):
                print ("Creating forward dynamics network: Using single network model")
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=model)
            else:
                print ("Creating forward dynamics network")
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None)
            forwardDynamicsModel.setActor(actor)
            forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
        
        (agent, learning_workers) = createLearningAgent(settings, output_experience_queue, state_bounds, action_bounds, reward_bounds)
        masterAgent = agent
        
        ### These are the workers for training
        (sim_workers, sim_work_queues) = createSimWorkers(settings, input_anchor_queue, output_experience_queue, eval_episode_data_queue, 
                                            model, forwardDynamicsModel, exp_val, state_bounds, action_bounds, reward_bounds)
        
        eval_sim_workers = sim_workers
        eval_sim_work_queues = sim_work_queues
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)): #True
            (eval_sim_workers, eval_sim_work_queues) = createSimWorkers(settings, input_anchor_queue_eval, output_experience_queue, 
                                                        eval_episode_data_queue, model, forwardDynamicsModel, exp_val, state_bounds, action_bounds, 
                                                        reward_bounds, default_sim_id=settings['override_sim_env_id']) # id=1
        else:
            input_anchor_queue_eval = input_anchor_queue
        

        best_eval=-100000000.0
        best_dynamicsLosses= best_eval*-1.0
            
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        dynamicsLosses = []
        dynamicsRewardLosses = []
        
        for lw in learning_workers:
            print ("Learning worker" )
            print (lw)
        
        if (int(settings["num_available_threads"]) > 1):
            for sw in sim_workers:
                print ("Sim worker")
                print (sw)
                sw.start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                for sw in eval_sim_workers:
                    print ("Sim worker")
                    print (sw)
                    sw.start()
        
        ## This needs to be done after the simulation worker processes are created
        exp_val = createEnvironment(settings["forwardDynamics_config_file"], settings['environment_type'], settings, render=settings['shouldRender'], index=0)
        exp_val.setActor(actor)
        exp_val.getActor().init()
        exp_val.init()
        
        ### This is for a single-threaded Synchronous sim only.
        if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
            sim_workers[0].setEnvironment(exp_val)
            sim_workers[0].start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                eval_sim_workers[0].setEnvironment(exp_val)
                eval_sim_workers[0].start()
        
        masterAgent.setPolicy(model)
        if (settings['train_forward_dynamics']):
            masterAgent.setForwardDynamics(forwardDynamicsModel)
        
        tmp_p=1.0
        message={}
        if (settings['load_saved_model']):
            tmp_p = settings['min_epsilon']
        data = ('Update_Policy', tmp_p, model.getStateBounds(), model.getActionBounds(), model.getRewardBounds(), 
                masterAgent.getPolicy().getNetworkParameters())
        if (settings['train_forward_dynamics']):
            data = ('Update_Policy', tmp_p, model.getStateBounds(), model.getActionBounds(), model.getRewardBounds(), 
                    masterAgent.getPolicy().getNetworkParameters(), masterAgent.getForwardDynamics().getNetworkParameters())
        message['type'] = 'Update_Policy'
        message['data'] = data
        for m_q in sim_work_queues:
            print("trainModel: Sending current network parameters: ", m_q)
            m_q.put(message)
        
        if ( int(settings["num_available_threads"]) ==  1):
           experience, state_bounds, reward_bounds, action_bounds = collectExperience(actor, exp_val, model, settings,
                           sim_work_queues=None, 
                           eval_episode_data_queue=None) #experience: state, action, nextstate, rewards, 
            
        else:
            if (settings['on_policy']):               
                experience, state_bounds, reward_bounds, action_bounds = collectExperience(actor, None, model, settings,
                           sim_work_queues=sim_work_queues, 
                           eval_episode_data_queue=eval_episode_data_queue)
            else:
                experience, state_bounds, reward_bounds, action_bounds = collectExperience(actor, None, model, settings,
                           sim_work_queues=input_anchor_queue, 
                           eval_episode_data_queue=eval_episode_data_queue)
        masterAgent.setExperience(experience)
        if ( 'keep_seperate_fd_exp_buffer' in settings and (settings['keep_seperate_fd_exp_buffer'])):
            masterAgent.setFDExperience(copy.deepcopy(experience))
        
        if (not validBounds(action_bounds)):
            # Check that the action bounds are spcified correctly
            print("Action bounds invalid: ", action_bounds)
            sys.exit()
        if (not validBounds(state_bounds)):
            # Probably did not collect enough bootstrapping samples to get good state bounds.
            print("State bounds invalid: ", state_bounds)
            state_bounds = fixBounds(np.array(state_bounds))
            bound_fixed = validBounds(state_bounds)
            print("State bounds fixed: ", bound_fixed)
        if (not validBounds(reward_bounds)):
            print("Reward bounds invalid: ", reward_bounds)
            sys.exit()
        
        print ("Reward History: ", experience._reward_history)
        print ("Action History: ", experience._action_history)
        print ("Action Mean: ", np.mean(experience._action_history))
        print ("Experience Samples: ", (experience.samples()))
        
        if (settings["save_experience_memory"]):
            print ("Saving initial experience memory")
            file_name=directory+getAgentName()+"_expBufferInit.hdf5"
            experience.saveToFile(file_name)

        if (settings['load_saved_model'] or (settings['load_saved_model'] == 'network_and_scales')): ## Transfer learning
            experience.setStateBounds(copy.deepcopy(model.getStateBounds()))
            experience.setRewardBounds(copy.deepcopy(model.getRewardBounds()))
            experience.setActionBounds(copy.deepcopy(model.getActionBounds()))
            model.setSettings(settings)
        else: ## Normal
            model.setStateBounds(state_bounds)
            model.setActionBounds(action_bounds)
            model.setRewardBounds(reward_bounds)
            experience.setStateBounds(copy.deepcopy(model.getStateBounds()))
            experience.setRewardBounds(copy.deepcopy(model.getRewardBounds()))
            experience.setActionBounds(copy.deepcopy(model.getActionBounds()))
                   
        masterAgent_message_queue = multiprocessing.Queue(settings['epochs'])
        
        if (settings['train_forward_dynamics']):
            if ( not settings['load_saved_model'] ):
                forwardDynamicsModel.setStateBounds(state_bounds)
                forwardDynamicsModel.setActionBounds(action_bounds)
                forwardDynamicsModel.setRewardBounds(reward_bounds)
            masterAgent.setForwardDynamics(forwardDynamicsModel)
        
        ## Now everything related to the exp memory needs to be updated
        bellman_errors=[]
        masterAgent.setPolicy(model)
        print("Master agent state bounds: ",  repr(masterAgent.getPolicy().getStateBounds()))
        for sw in sim_workers: # Need to update parameter bounds for models
            print ("exp: ", sw._exp)
            print ("sw modle: ", sw._model.getPolicy()) 
            
        ## If not on policy
        if ( not settings['on_policy']):
            for lw in learning_workers:
                lw._agent.setPolicy(model)
                lw.setMasterAgentMessageQueue(masterAgent_message_queue)
                lw.updateExperience(experience)
                print ("ls policy: ", lw._agent.getPolicy())
                
                lw.start()
            
        tmp_p=1.0
        if ( settings['load_saved_model'] ):
            tmp_p = settings['min_epsilon']
        data = ('Update_Policy', tmp_p, model.getStateBounds(), model.getActionBounds(), model.getRewardBounds(), 
                masterAgent.getPolicy().getNetworkParameters())
        if (settings['train_forward_dynamics']):
            data = ('Update_Policy', tmp_p, model.getStateBounds(), model.getActionBounds(), model.getRewardBounds(), 
                    masterAgent.getPolicy().getNetworkParameters(), masterAgent.getForwardDynamics().getNetworkParameters())
        message['type'] = 'Update_Policy'
        message['data'] = data
        for m_q in sim_work_queues:
            print("trainModel: Sending current network parameters: ", m_q)
            m_q.put(message)
            
        del model
        ## Give gloabl access to processes to they can be terminated when ctrl+c is pressed
        global sim_processes
        sim_processes = sim_workers
        global learning_processes
        learning_processes = learning_workers
        global _input_anchor_queue
        _input_anchor_queue = input_anchor_queue
        global _output_experience_queue
        _output_experience_queue = output_experience_queue
        global _eval_episode_data_queue
        _eval_episode_data_queue = eval_episode_data_queue
        global _sim_work_queues
        _sim_work_queues = sim_work_queues
            
        trainData = {}
        trainData["mean_reward"]=[]
        trainData["std_reward"]=[]
        trainData["mean_bellman_error"]=[]
        trainData["std_bellman_error"]=[]
        trainData["mean_discount_error"]=[]
        trainData["std_discount_error"]=[]
        trainData["mean_forward_dynamics_loss"]=[]
        trainData["std_forward_dynamics_loss"]=[]
        trainData["mean_forward_dynamics_reward_loss"]=[]
        trainData["std_forward_dynamics_reward_loss"]=[]
        trainData["mean_eval"]=[]
        trainData["std_eval"]=[]
        trainData["mean_critic_loss"]=[]
        trainData["std_critic_loss"]=[]
        trainData["mean_critic_regularization_cost"]=[]
        trainData["std_critic_regularization_cost"]=[]
        trainData["mean_actor_loss"]=[]
        trainData["std_actor_loss"]=[]
        trainData["mean_actor_regularization_cost"]=[]
        trainData["std_actor_regularization_cost"]=[]
        trainData["anneal_p"]=[]
        
        if (False):
            print("State Bounds:", masterAgent.getStateBounds())
            print("Action Bounds:", masterAgent.getActionBounds())
            
            print("Exp State Bounds: ", experience.getStateBounds())
            print("Exp Action Bounds: ", experience.getActionBounds())
        
        print ("Starting first round")
        if (settings['on_policy']):
            sim_epochs_ = epochs
        for round_ in range(0,rounds): #annel value # the parameter of greedy exploration
            if ('annealing_schedule' in settings and (settings['annealing_schedule'] != False)):
                p = anneal_value(float(round_/rounds), settings_=settings)
            else:
                p = ((settings['initial_temperature']/math.log(round_+2))) 
            p = max(settings['min_epsilon'], min(settings['epsilon'], p)) # Keeps it between 1.0 and 0.2
            if ( settings['load_saved_model'] ):
                p = settings['min_epsilon']
                
            for epoch in range(epochs):
                if (settings['on_policy']):
                    
                    out = simModelParrallel(sw_message_queues=sim_work_queues,
                                            model=masterAgent, settings=settings, 
                                            eval_episode_data_queue=eval_episode_data_queue, 
                                            anchors=settings['num_on_policy_rollouts'])
                    (tuples, discounted_sum, q_value, evalData) = out  # tuples = states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions
                    (__states, __actions, __result_states, __rewards, __falls, __G_ts, advantage__, exp_actions__) = tuples
                    for i in range(1):
                        masterAgent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states,
                                           _falls=__falls, _advantage=advantage__, _exp_actions=exp_actions__)
                    
                    if (('anneal_on_policy' in settings) and settings['anneal_on_policy']):  
                        p_tmp_ = p 
                    else:
                        p_tmp_ = 1.0
                    data = ('Update_Policy', p_tmp_, 
                            masterAgent.getStateBounds(),
                            masterAgent.getActionBounds(),
                            masterAgent.getRewardBounds(),
                            masterAgent.getPolicy().getNetworkParameters())
                    message = {}
                    message['type'] = 'Update_Policy'
                    message['data'] = data
                    if (settings['train_forward_dynamics']):
                        data = ('Update_Policy', p_tmp_, 
                                masterAgent.getStateBounds(),
                                masterAgent.getActionBounds(),
                                masterAgent.getRewardBounds(),
                                masterAgent.getPolicy().getNetworkParameters(),
                                masterAgent.getForwardDynamics().getNetworkParameters())
                        message['data'] = data
                    for m_q in sim_work_queues:
                        ## block on full queue
                        m_q.put(message)
                    
                    if ('override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                        for m_q in eval_sim_work_queues:
                            ## block on full queue
                            m_q.put(message)
    
                else:
                    episodeData = {}
                    episodeData['data'] = epoch
                    episodeData['type'] = 'sim'
                    input_anchor_queue.put(episodeData)
                
                if masterAgent.getExperience().samples() >= batch_size: #更新policy网络
                    states, actions, result_states, rewards, falls, G_ts, exp_actions = masterAgent.getExperience().get_batch(batch_size)
                    error = masterAgent.bellman_error(states, actions, rewards, result_states, falls)                    
                    bellman_errors.append(error)
                    if (settings['debug_critic']):
                        loss__ = masterAgent.getPolicy()._get_critic_loss() # uses previous call batch data
                        criticLosses.append(loss__)
                        regularizationCost__ = masterAgent.getPolicy()._get_critic_regularization()
                        criticRegularizationCosts.append(regularizationCost__)
                        
                    if (settings['debug_actor']): #True
                        loss__ = masterAgent.getPolicy()._get_actor_loss() # uses previous call batch data
                        actorLosses.append(loss__)
                        regularizationCost__ = masterAgent.getPolicy()._get_actor_regularization()
                        actorRegularizationCosts.append(regularizationCost__)
                    
                    if not all(np.isfinite(error)):
                        print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions) + " Falls: ", str(falls))
                        print ("Bellman Error is Nan: " + str(error) + str(np.isfinite(error)))
                        sys.exit()
                    
                    error = np.mean(np.fabs(error))
                    if error > 10000:
                        print ("Error to big: ")
                        print (states, actions, rewards, result_states)
                        
                    if (settings['train_forward_dynamics']): #False
                        dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(states, actions, result_states, rewards)
                        dynamicsLoss = np.mean(np.fabs(dynamicsLoss)) #fabs：计算绝对值
                        dynamicsLosses.append(dynamicsLoss)
                        if (settings['train_reward_predictor']):
                            dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(states, actions, result_states, rewards)
                            dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss))
                            dynamicsRewardLosses.append(dynamicsRewardLoss)
                    if (settings['train_forward_dynamics']):
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss))
                    else:
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error))
                    
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("Master agent experience size: " + str(masterAgent.getExperience().samples()))                
                
                if (not settings['on_policy']):
                    ## There could be stale policy parameters in here, use the last set put in the queue
                    data = None
                    while (not masterAgent_message_queue.empty()):
                        ## Don't block
                        try:
                            data = masterAgent_message_queue.get(False)
                        except Exception as inst:                        
                            pass
                    if (not (data == None) ):                   
                        masterAgent.setExperience(data[0])
                        masterAgent.getPolicy().setNetworkParameters(data[1])
                        masterAgent.setStateBounds(masterAgent.getExperience().getStateBounds())
                        masterAgent.setActionBounds(masterAgent.getExperience().getActionBounds())
                        masterAgent.setRewardBounds(masterAgent.getExperience().getRewardBounds())
                        if (settings['train_forward_dynamics']):
                            masterAgent.getForwardDynamics().setNetworkParameters(data[2])
                            if ( 'keep_seperate_fd_exp_buffer' in settings and (settings['keep_seperate_fd_exp_buffer'])):
                                masterAgent.setFDExperience(data[3])

                # this->_actor->iterate();
            ## This will let me know which part of learning is going slower training updates or simulation
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                print ("sim queue size: ", input_anchor_queue.qsize()) #返回队列的大小
            if ( output_experience_queue != None):
                print ("exp tuple queue size: ", output_experience_queue.qsize())
            
            if (not settings['on_policy']):
                data = ('Update_Policy', p,
                        masterAgent.getStateBounds(),
                        masterAgent.getActionBounds(),
                        masterAgent.getRewardBounds(),
                        masterAgent.getPolicy().getNetworkParameters())
                if (settings['train_forward_dynamics']):
                    data = ('Update_Policy', p, 
                            masterAgent.getStateBounds(),
                            masterAgent.getActionBounds(),
                            masterAgent.getRewardBounds(),
                            masterAgent.getPolicy().getNetworkParameters(),
                            masterAgent.getForwardDynamics().getNetworkParameters())
                message['type'] = 'Update_Policy'
                message['data'] = data
                for m_q in sim_work_queues:
                    ## Don't block on full queue
                    try:
                        m_q.put(message, False)
                    except: 
                        print ("SimWorker model parameter message queue full: ", m_q.qsize())
                if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                    for m_q in eval_sim_work_queues:
                        ## Don't block on full queue
                        try:
                            m_q.put(message, False)
                        except: 
                            print ("SimWorker model parameter message queue full: ", m_q.qsize())
              
            if (round_ % settings['plotting_update_freq_num_rounds']) == 0:
                # Running less often helps speed learning up.
                # Sync up sim actors
                if (settings['on_policy']):
                    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModelParrallel( input_anchor_queue=eval_sim_work_queues,
                                                               model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
                else:
                    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModelParrallel( input_anchor_queue=input_anchor_queue_eval,
                                                               model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])

                print (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error)
                if mean_bellman_error > 10000:
                    print ("Error to big: ")
                else:
                    if (settings['train_forward_dynamics']): #false
                        mean_dynamicsLosses = np.mean(dynamicsLosses)
                        std_dynamicsLosses = np.std(dynamicsLosses)
                        dynamicsLosses = []
                    if (settings['train_reward_predictor']): #false
                        mean_dynamicsRewardLosses = np.mean(dynamicsRewardLosses)
                        std_dynamicsRewardLosses = np.std(dynamicsRewardLosses)
                        dynamicsRewardLosses = []
                        
                    trainData["mean_reward"].append(mean_reward)
                    trainData["std_reward"].append(std_reward)
                    trainData["anneal_p"].append(p)
                    trainData["mean_bellman_error"].append(np.mean(np.fabs(bellman_errors)))
                    trainData["std_bellman_error"].append(np.std(bellman_errors))
                    bellman_errors=[]
                    trainData["mean_discount_error"].append(mean_discount_error)
                    trainData["std_discount_error"].append(std_discount_error)
                    trainData["mean_eval"].append(mean_eval)
                    trainData["std_eval"].append(std_eval)
                    if (settings['train_forward_dynamics']):
                        trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
                        trainData["std_forward_dynamics_loss"].append(std_dynamicsLosses)
                    if (settings['train_reward_predictor']):
                        trainData["mean_forward_dynamics_reward_loss"].append(mean_dynamicsRewardLosses)
                        trainData["std_forward_dynamics_reward_loss"].append(std_dynamicsRewardLosses)              
                
            if (round_ % settings['saving_update_freq_num_rounds']) == 0:            
                if (settings['train_forward_dynamics']):
                    file_name_dynamics=directory+"forward_dynamics_"+".pkl"
                    f = open(file_name_dynamics, 'wb')
                    dill.dump(masterAgent.getForwardDynamics(), f)
                    f.close()
                    if mean_dynamicsLosses < best_dynamicsLosses:
                        best_dynamicsLosses = mean_dynamicsLosses
                        print ("Saving BEST current forward dynamics agent: " + str(best_dynamicsLosses))
                        file_name_dynamics=directory+"forward_dynamics_"+"_Best.pkl"
                        f = open(file_name_dynamics, 'wb')
                        dill.dump(masterAgent.getForwardDynamics(), f) #save model
                        f.close()
                        
                if (mean_eval > best_eval):
                    best_eval = mean_eval
                    print ("Saving BEST current agent: " + str(best_eval))
                    file_name=directory+getAgentName()+"_Best.pkl"
                    f = open(file_name, 'wb')
                    dill.dump(masterAgent.getPolicy(), f)
                    f.close()
                    
                if settings['save_trainData']:
                    fp = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", 'w')
                    ## because json does not serialize np.float32 
                    for key in trainData:
                        trainData[key] = [float(i) for i in trainData[key]]
                    json.dump(trainData, fp)
                    fp.close()
                        
                print ("Saving current masterAgent")
                
                file_name=directory+getAgentName()+".pkl"
                f = open(file_name, 'wb')
                dill.dump(masterAgent.getPolicy(), f)
                f.close()
                
            gc.collect()    

        print ("Terminating Workers")
        if (settings['on_policy']):
            for m_q in sim_work_queues:
                ## block on full queue
                m_q.put(None)
            if ('override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                for m_q in eval_sim_work_queues:
                    ## block on full queue
                    m_q.put(None)
            for sw in sim_workers: # Should update these more often
                sw.join()
            if ('override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                for sw in eval_sim_workers: # Should update these more often
                    sw.join() 
        
        for i in range(len(sim_work_queues)):
            print ("sim_work_queues size: ", sim_work_queues[i].qsize())
            while (not sim_work_queues[i].empty()): ### Empty the queue
                ## Don't block
                try:
                    data_ = sim_work_queues[i].get(False)
                except Exception as inst:
                    pass
            print ("sim_work_queues size: ", sim_work_queues[i].qsize())
            
            
        for i in range(len(eval_sim_work_queues)):
            print ("eval_sim_work_queues size: ", eval_sim_work_queues[i].qsize())
            while (not eval_sim_work_queues[i].empty()): ### Empty the queue
                ## Don't block
                try:
                    data_ = eval_sim_work_queues[i].get(False)
                except Exception as inst:
                    pass
            print ("eval_sim_work_queues size: ", eval_sim_work_queues[i].qsize())
        
        
        print ("Finish sim")
        exp_val.finish()
        
        print ("Save last versions of files.")
        file_name=directory+getAgentName()+".pkl"
        f = open(file_name, 'wb')
        dill.dump(masterAgent.getPolicy(), f)
        f.close()
        
        f = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", "w")
        for key in trainData:
            trainData[key] = [float(i) for i in trainData[key]]
        json.dump(trainData, f, sort_keys=True, indent=4)
        f.close()
        
        if (settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+".pkl"
            f = open(file_name_dynamics, 'wb')
            dill.dump(masterAgent.getForwardDynamics(), f)
            f.close()
       
        print("Delete any plots being used")

        gc.collect() #立即释放内存
        
import inspect
def print_full_stack(tb=None):
    """
    Only good way to print stack trace yourself.
    http://blog.dscpl.com.au/2015/03/generating-full-stack-traces-for.html
    """
    if tb is None:
        tb = sys.exc_info()[2]

    print ('Traceback (most recent call last):')
    if (not (tb == None)):
        for item in reversed(inspect.getouterframes(tb.tb_frame)[1:]):
            print (' File "{1}", line {2}, in {3}\n'.format(*item),)
            for line in item[4]:
                print (' ' + line.lstrip(),)
            for item in inspect.getinnerframes(tb):
                print (' File "{1}", line {2}, in {3}\n'.format(*item),)
            for line in item[4]:
                print (' ' + line.lstrip(),)
            
import signal
import sys
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        print("sim processes: ", sim_processes)
        print("learning_processes: ", learning_processes)
        
        ## cancel all the queues
        _input_anchor_queue.cancel_join_thread()
        _output_experience_queue.cancel_join_thread()
        _eval_episode_data_queue.cancel_join_thread()
        for sim_queue in _sim_work_queues:
            sim_queue.cancel_join_thread()
        
        
        for proc in sim_processes:
            if (not (proc == None)):
                print ("Killing process: ", proc)
                print ("process id: ", proc.pid())
                os.kill(proc.pid(), signal.SIGINT)
        for proc in learning_processes:
            if (not (proc == None)):
                print ("Killing process: ", proc.pid())
                os.kill(proc.pid(), signal.SIGINT)
            
        print_full_stack()
        sys.exit(0)

if (__name__ == "__main__"):
    
    """
        python trainModel.py <sim_settings_file>
        Example:
        python trainModel.py settings/navGame/PPO_5D.json 
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
        
    t0 = time.time()
    trainModelParallel((sys.argv[1], settings))
    t1 = time.time()
    print ("Model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")


    print("All Done.")