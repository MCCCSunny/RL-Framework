import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
import lasagne
import sys
import copy
import dill
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty
from util.SimulationUtil import getAgentName
import pdb
class Distillation(AlgorithmInterface):
    '''
    self._model: original model
    self._modelTarget:要更新的模型
    '''
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(Distillation,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        ### Load expert policy files
        self._expert_policies = []
        file_name_ = ""
        for i in range(len(self.getSettings()['expert_policy_files'])):
            file_name = self.getSettings()['expert_policy_files'][i] + '/'+ self.getSettings()['model_type']+'/'+getAgentName()+'.pkl'
            if (file_name_ == file_name):
                ## To help save memory when experts are the same
                self._expert_policies.append(model_)
            else:
                print ("Loading pre compiled network: ", file_name)
                f = open(file_name, 'rb')
                model_ = dill.load(f)
                f.close()
                self._expert_policies.append(model_) # expert model, load the 2 expert models 
            file_name_ = file_name
            
        self._actor_buffer_states=[]
        self._actor_buffer_result_states=[]
        self._actor_buffer_actions=[]
        self._actor_buffer_rewards=[]
        self._actor_buffer_falls=[]
        self._actor_buffer_diff=[]
        
        self._NotFallen = T.bcol("Not_Fallen")
        ## because float64 <= float32 * int32, need to use int16 or int8
        self._NotFallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._NotFallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))
        
        self._tmp_diff = T.col("Tmp_Diff")
        self._tmp_diff.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        
        self._tmp_diff_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True)) #定义一个共享变量，初始值为为0

        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        self._critic_learning_rate = self.getSettings()["critic_learning_rate"]
        ## Target network
        self._modelTarget = copy.deepcopy(model)# target model 是要更新的模型
        
        self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True) #确定性原始模型的state值输出
        self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False) #非确定的state值输出
        self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True) #下一步的state值
        self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True) #目标模型的下一步的state值
        self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True) #目标模型的state值
        self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False) #目标模型的state
        
        self._q_valsActA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True) #remove the random 
        self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False) #actor 值
        
        self._q_func = self._q_valsA
        self._q_funcTarget = self._q_valsTarget
        self._q_func_drop = self._q_valsA_drop
        self._q_funcTarget_drop = self._q_valsTarget_drop
        self._q_funcAct = self._q_valsActA
        self._q_funcAct_drop = self._q_valsActA_drop
        
        self._target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState) 
        # self._model.getRewardSymbolicVariable() 获取rewards的值getRewards() =self._rewards_shared 从0开始一直更新
        self._diff = self._target - self._q_func
        self._diff_drop = self._target - self._q_func_drop #更新的模型的reward减去原始模型的critic的输出值
        loss = T.pow(self._diff, 2) 
        self._loss = T.mean(loss) # 两个模型的reward的差值
        self._loss_drop = T.mean(0.5 * self._diff_drop ** 2)
        
        self._params = lasagne.layers.helper.get_all_params(self._model.getCriticNetwork())
        self._actionParams = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards()
        }
        self._actGivens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._tmp_diff: self._tmp_diff_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * lasagne.regularization.regularize_network_params(
                                        self._model.getCriticNetwork(), lasagne.regularization.l2))
        self._actor_regularization = ( (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getActorNetwork(), lasagne.regularization.l2)) )
        if (self.getSettings()['use_previous_value_regularization']):
            self._actor_regularization = self._actor_regularization + (( self.getSettings()['previous_value_regularization_weight']) * 
                       change_penalty(self._model.getActorNetwork(), self._modelTarget.getActorNetwork()))
        elif ('regularization_type' in self.getSettings() and (self.getSettings()['regularization_type'] == 'KL_Divergence')):
            self._kl_firstfixed = T.mean(kl(self._q_valsActTarget, T.ones_like(self._q_valsActTarget) * self.getSettings()['exploration_rate'], self._q_valsActA, T.ones_like(self._q_valsActA) * self.getSettings()['exploration_rate'], self._action_length))
            self._actor_regularization = (self._kl_firstfixed) *(self.getSettings()['kl_divergence_threshold'])
            
            print("Using regularization type : ", self.getSettings()['regularization_type']) 
        # SGD update
        self._value_grad = T.grad(self._loss + self._critic_regularization
                                                     , self._params)
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.rmsprop(self._value_grad, self._params, self._learning_rate, self._rho,
                                           self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.momentum(self._value_grad
                                                      , self._params, self._critic_learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adam(self._value_grad
                        , self._params, self._critic_learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adagrad(self._value_grad
                        , self._params, self._critic_learning_rate, epsilon=self._rms_epsilon)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            sys.exit(-1)
        ## TD update

        ## Need to perform an element wise operation or replicate _diff for this to work properly.
        self._actDiff = (self._model.getActionSymbolicVariable() - self._q_valsActA_drop)  # 更新模型的actor的输出减去原始模型的actor值
                                                                                
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((T.mean(T.pow(self._actDiff, 2),axis=1)), 
                                                                                (self._tmp_diff))
        self._actLoss = T.mean(self._actLoss_) 
        self._policy_grad = T.grad(self._actLoss + self._actor_regularization ,  self._actionParams)
        ## Clipping the max gradient
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._actionUpdates = lasagne.updates.rmsprop(self._policy_grad, self._actionParams, 
                    self._learning_rate , self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._actionUpdates = lasagne.updates.momentum(self._policy_grad, self._actionParams, 
                    self._learning_rate , momentum=self._rho)
        elif (self.getSettings()['optimizer'] == 'adam'):
            self._actionUpdates = lasagne.updates.adam(self._policy_grad, self._actionParams, 
                    self._learning_rate , beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'adagrad'):
            self._actionUpdates = lasagne.updates.adagrad(self._policy_grad, self._actionParams, 
                    self._learning_rate, epsilon=self._rms_epsilon)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            
        self._givens_grad = {self._model.getStateSymbolicVariable(): self._model.getStates()}

               
        ## Bellman error
        self._bellman = self._target - self._q_funcTarget
        
        ### Give v(s') the next state and v(s) (target) the current state  
        self._diff_adv = (self._discount_factor * self._q_func) - (self._q_valsTargetNextState )
        self._diff_adv_givens = {
            self._model.getStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getStates(),
        }
        
        Distillation.compile(self)
        
    def compile(self):
        
        #### Stuff for Debugging #####
        self._get_diff = theano.function([], [self._diff], givens=self._givens_)
        self._get_target = theano.function([], [self._target], givens={
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards()
        })
        ## Always want this one
        self._get_critic_loss = theano.function([], [self._loss], givens=self._givens_)
        if (self.getSettings()['debug_critic']):
            self._get_critic_regularization = theano.function([], [self._critic_regularization])
        
        if (self.getSettings()['debug_actor']):
            if (self.getSettings()['regularization_type'] == 'KL_Divergence'):
                self._get_actor_regularization = theano.function([], [self._actor_regularization], 
                                                                 givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
            else:
                self._get_actor_regularization = theano.function([], [self._actor_regularization])
            self._get_actor_loss = theano.function([], [self._actLoss], givens=self._actGivens)
            self._get_actor_diff_ = theano.function([], [self._actDiff], givens={
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                self._model.getActionSymbolicVariable(): self._model.getActions()
            }) 
       
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        self._trainActor = theano.function([], [self._actLoss, self._q_func_drop], updates=self._actionUpdates, givens=self._actGivens)
        self._q_val = theano.function([], self._q_func,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._val_TargetState = theano.function([], self._q_funcTarget,
                                       givens={self._model.getStateSymbolicVariable(): self._modelTarget.getStates()})
        self.get_q_valsTargetNextState = theano.function([], self._q_valsTargetNextState,
                                       givens={self._model.getResultStateSymbolicVariable(): self._model.getResultStates()})
        

        self._q_action = theano.function([], self._q_valsActA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._bellman_error2 = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        if ( 'optimize_advantage_for_MBAE' in self.getSettings() and  self.getSettings()['optimize_advantage_for_MBAE'] ):
            self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._diff_adv), 
                            [self._model._stateInputVar] + self._params), allow_input_downcast=True, givens=self._diff_adv_givens)
        else:
            self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._q_func), 
                            [self._model._stateInputVar] + self._params), allow_input_downcast=True, givens=self._givens_grad)
            
        self._get_grad_policy = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._actLoss, 
                            [self._model._stateInputVar] + self._actionParams), allow_input_downcast=True, givens=self._actGivens)
        
    def updateTargetModel(self): #student model
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        if (( 'lerp_target_network' in self.getSettings()) and self.getSettings()['lerp_target_network']) :
            all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork()) #
            all_paramsB = lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork())
            lerp_weight = 0.01
            
            all_params = []
            for paramsA, paramsB in zip(all_paramsA, all_paramsB):
                params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
                all_params.append(params)
                
            lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), all_params)
        else:
            all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork())
            all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
            lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), all_paramsA)
            lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA) 
    
    def setData(self, states, actions, rewards, result_states, fallen):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        self._model.setRewards(rewards)
        self._modelTarget.setStates(states)
        self._modelTarget.setResultStates(result_states)
        self._modelTarget.setActions(actions)
        self._modelTarget.setRewards(rewards)
        self._NotFallen_shared.set_value(fallen)
        ## Easy fix for computing actor loss
        diff = self._bellman_error2()
        self._tmp_diff_shared.set_value(diff)

        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        print (falls,'-------------------------------------------------')
        pdb.set_trace()
        self.setData(states, actions, rewards, result_states, falls)
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        loss = 0
        pre_loss = self._get_critic_loss()[0]
        if ( pre_loss < 10.0): ## To protect the critic from odd losses
            loss, _ = self._train()
        
        return loss # loss=(reward-criticvalue)**2
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, forwardDynamicsModel=None):
        lossActor = 0
        print (falls,'-------------------------------------------------')
        pdb.set_trace()
        ### Update actions to expert actions. Some were selected from current policy

        if ('run_distillation_in_test_mode' in self.getSettings() and (self.getSettings()['run_distillation_in_test_mode'])): 
            #in the test mode, the agent so not use the distill
            pass
        else:        
            actions_ = []
            for i in range(states.shape[0]):
                expert_index = falls[i][0]
                state_ = [states[i]]
                ### Need to convert normalized state back to env scaled state
                state_ = scale_action(state_, self.getStateBounds())
                action_ = self._expert_policies[expert_index].predict(state_) # expert result
                action_ = norm_state(action_, self.getActionBounds())
                actions_.append(action_)
            actions_ = np.array(actions_, dtype=self.getSettings()['float_type'])
        actions = actions_ #2 experts's actions
        # here, the state is the input value, and the action is the 2 experts' actions, 
        # 也就是，expert只提供了action值，其他的还是采用原来的
        for i in range(states.shape[0]): ### Put data in buffer 
            self._actor_buffer_diff.append([1.0])
            self._actor_buffer_states.append(states[i])
            self._actor_buffer_actions.append(actions[i])
            self._actor_buffer_rewards.append(rewards[i])
            self._actor_buffer_result_states.append(result_states[i])
            self._actor_buffer_falls.append(falls[i])
        while (len(self._actor_buffer_diff) > self.getSettings()['batch_size']):
            ### Get batch from buffer
            tmp_states = self._actor_buffer_states[:self.getSettings()['batch_size']]
            tmp_actions = self._actor_buffer_actions[:self.getSettings()['batch_size']]
            tmp_rewards = self._actor_buffer_rewards[:self.getSettings()['batch_size']]
            tmp_result_states = self._actor_buffer_result_states[:self.getSettings()['batch_size']]
            tmp_falls = self._actor_buffer_falls[:self.getSettings()['batch_size']]
            tmp_diff = np.array(self._actor_buffer_diff[:self.getSettings()['batch_size']], dtype=self.getSettings()['float_type'])
            self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
            self._tmp_diff_shared.set_value(tmp_diff)
            lossActor, _ = self._trainActor()
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor, " actor buffer size: ", len(self._actor_buffer_actions))
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):    
                actions_ = self._q_action()
                print("Mean action: ", np.mean(actions_, axis=0), " std ", np.std(actions_, axis=0))
            ### Remove batch from buffer
            self._actor_buffer_states=self._actor_buffer_states[self.getSettings()['batch_size']:]
            self._actor_buffer_actions = self._actor_buffer_actions[self.getSettings()['batch_size']:]
            self._actor_buffer_rewards = self._actor_buffer_rewards[self.getSettings()['batch_size']:]
            self._actor_buffer_result_states = self._actor_buffer_result_states[self.getSettings()['batch_size']:]
            self._actor_buffer_falls =self._actor_buffer_falls[self.getSettings()['batch_size']:]
            self._actor_buffer_diff = self._actor_buffer_diff[self.getSettings()['batch_size']:]
        return lossActor
    
    def trainDyna(self, predicted_states, actions, rewards, result_states, falls):
        """
            Performs a DYNA type update
            Because I am using target network a direct DYNA update does nothing. 
            The gradients are not calculated for the target network.
            L(\theta) = (r + V(s'|\theta')) - V(s|\theta))
            Instead what is done is this
            L(\theta) = V(s_1|\theta')) - V(s_2|\theta))
            Where s1 comes from the simulation and s2 is a predicted and noisey value from an fd model
            Parameters
            ----------
            predicted_states : predicted states, s_1
            
            actions : list of actions
                
            rewards : rewards for taking action a_i
            
            result_states : simulated states, s_2
            
            falls: list of flags for whether or not the character fell
            Returns
            -------
            
            loss: the loss for the DYNA type update

        """
        print (falls,'-------------------------------------------------')
        pdb.set_trace()
        return 0
    
    def train(self, states, actions, rewards, result_states, falls):
        print (falls,'-------------------------------------------------')
        pdb.set_trace()
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss

    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, 
                bootstrapping=False):
        if (not (p is None) and (evaluation_ is False)):
            r = np.random.rand(1)[0] ## in [0,1]
            if ('run_distillation_in_test_mode' in self.getSettings() and (self.getSettings()['run_distillation_in_test_mode'])):
                pass
            else:
                r = 0.0 ### Fix for debugging, expert only
            if ( r > p):
                evaluation_ = True
        ### Want to start out selecting actions from the expert more
        ### p starts at 1 is anneal to 0.
        if ((evaluation_ is True) or (bootstrapping is True)): ## Use policy
            # print("Using Policy")
            action_ = super(Distillation,self).predict(state)
        else: ## Use expert policy  
            action_ = self._expert_policies[sim_index].predict(state)
        return action_
    