

"""
    fitModelToData
    This script is designed particularly for debugging.
    It will attempt to fit a RL model to the given data, no simulation will be done.
    This should help debug any issues with the model and the optimization/fitting of the model.

"""
"""
theano.config.device='gpu'
theano.config.mode='FAST_RUN'
theano.config.floatX='float32'
"""

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

def fitModelToData(settingsFileName):
    """
    State is the input state and Action is the desired output (y).
    """
    # from model.ModelUtil import *
    
    file = open(settingsFileName)
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings)))
    file.close()
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience, simEpoch, evalModel
    from model.ModelUtil import validBounds
    from model.LearningAgent import LearningAgent, LearningWorker
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    from sim.PendulumEnvState import PendulumEnvState
    from sim.PendulumEnv import PendulumEnv
    from sim.BallGame2DEnv import BallGame2DEnv 
    import time  
    
    settings = validateSettings(settings)

    train_forward_dynamics=True
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0] # number of rows
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    reward_bounds=np.array(settings["reward_bounds"])
    batch_size=settings["batch_size"]
    train_on_validation_set=settings["train_on_validation_set"]
    state_bounds = np.array(settings['state_bounds'])
    discrete_actions = np.array(settings['discrete_actions'])
    print ("Sim config file name: ", str(settings["sim_config_file"]))
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    file_name=directory+getAgentName()+"expBufferInit.hdf5"
    experience.loadFromFile(file_name)
    state_bounds = experience._state_bounds
    action_bounds = experience._action_bounds
    reward_bounds = experience._reward_bounds
    
    output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    learning_workers = []
    for process in range(1):
        # this is the process that selects which game to play
        agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        
        agent.setSettings(settings)
        
        lw = LearningWorker(output_experience_queue, agent, namespace)
        learning_workers.append(lw)  
    masterAgent = agent
    masterAgent.setExperience(experience)
    
    if action_space_continuous:
        model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
    else:
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    if ( not settings['load_saved_model'] ):
        model.setStateBounds(state_bounds)
        model.setActionBounds(action_bounds)
        model.setRewardBounds(reward_bounds)
    else: # continuation learning
        experience.setStateBounds(model.getStateBounds())
        experience.setRewardBounds(model.getRewardBounds())
        experience.setActionBounds(model.getActionBounds())
        
    
    if (settings['train_forward_dynamics']):
        print ("Created forward dynamics network")
        forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)        
        forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
        namespace.forwardNN = masterAgent.getForwardDynamics().getNetworkParameters()
        namespace.forwardDynamicsModel = forwardDynamicsModel
    
    ## Now everything related to the exp memory needs to be updated
    bellman_errors=[]
    masterAgent.setPolicy(model)
    namespace.agentPoly = masterAgent.getPolicy().getNetworkParameters()
    namespace.model = model
    
    
    if (settings['visualize_learning']):
        rlv = NNVisualize(title=str(directory), settings=settings)
        rlv.setInteractive()
        rlv.init()
            
    if (settings['debug_critic']):
        criticLosses = []
        criticRegularizationCosts = [] 
        if (settings['visualize_learning']):
            critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + str(settings["model_type"]))
            critic_loss_viz.setInteractive()
            critic_loss_viz.init()
            critic_regularization_viz = NNVisualize(title=str("Critic Regularization Cost") + " with " + str(settings["model_type"]))
            critic_regularization_viz.setInteractive()
            critic_regularization_viz.init()
        
    if (settings['debug_actor']):
        actorLosses = []
        actorRegularizationCosts = []            
        if (settings['visualize_learning']):
            actor_loss_viz = NNVisualize(title=str("Actor Loss") + " with " + str(settings["model_type"]))
            actor_loss_viz.setInteractive()
            actor_loss_viz.init()
            actor_regularization_viz = NNVisualize(title=str("Actor Regularization Cost") + " with " + str(settings["model_type"]))
            actor_regularization_viz.setInteractive()
            actor_regularization_viz.init()
                
    trainData = {}
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    trainData["mean_forward_dynamics_loss"]=[]
    trainData["std_forward_dynamics_loss"]=[]
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
        
    best_dynamicsLosses=1000000
    _states, _actions, _result_states, _rewards, _falls, _G_ts = experience.get_batch(batch_size)
    for round_ in range(rounds):
        t0 = time.time()
        __states, __actions, __result_states, __rewards, __falls, __G_ts = experience.get_batch(100)
        for i in range(1):
            masterAgent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states, _falls=__falls)
        t1 = time.time()
        time_taken = t1 - t0
        if masterAgent.getExperience().samples() > batch_size:
            states, actions, result_states, rewards, falls, G_ts = masterAgent.getExperience().get_batch(batch_size)
            print ("Batch size: " + str(batch_size))
            error = masterAgent.bellman_error(states, actions, rewards, result_states, falls)
            bellman_errors.append(error)
            if (settings['debug_critic']):
                loss__ = masterAgent.getPolicy()._get_critic_loss() # uses previous call batch data
                criticLosses.append(loss__)
                regularizationCost__ = masterAgent.getPolicy()._get_critic_regularization()
                criticRegularizationCosts.append(regularizationCost__)
                
            if (settings['debug_actor']):
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
                
            if (settings['train_forward_dynamics']):
                dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(states, actions, result_states, rewards)
                dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
                dynamicsLosses.append(dynamicsLoss)
            if (settings['train_forward_dynamics']):
                print ("Round: " + str(round_) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss) + " in " + str(time_taken) + " seconds")
            else:
                print ("Round: " + str(round_) + " bellman error: " + str(error) + " in " + str(time_taken) + " seconds")
           

        print ("Master agent experience size: " + str(masterAgent.getExperience().samples()))
        trainData["mean_bellman_error"].append(np.mean(np.fabs(bellman_errors)))
        trainData["std_bellman_error"].append(np.std(bellman_errors))
        if (settings['visualize_learning']):
            rlv.updateLoss(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
            rlv.redraw()
            rlv.setInteractiveOff()
            rlv.saveVisual(directory+"trainingGraphNN")
            rlv.setInteractive()
        # print "Error: " + str(error)
        if (settings['debug_critic']):
            mean_criticLosses = np.mean(criticLosses)
            std_criticLosses = np.std(criticLosses)
            trainData["mean_critic_loss"].append(mean_criticLosses)
            trainData["std_critic_loss"].append(std_criticLosses)
            criticLosses = []
            if (settings['visualize_learning']):
                critic_loss_viz.updateLoss(np.array(trainData["mean_critic_loss"]), np.array(trainData["std_critic_loss"]))
                critic_loss_viz.redraw()
                critic_loss_viz.setInteractiveOff()
                critic_loss_viz.saveVisual(directory+"criticLossGraph")
                critic_loss_viz.setInteractive()
            
            mean_criticRegularizationCosts = np.mean(criticRegularizationCosts)
            std_criticRegularizationCosts = np.std(criticRegularizationCosts)
            trainData["mean_critic_regularization_cost"].append(mean_criticRegularizationCosts)
            trainData["std_critic_regularization_cost"].append(std_criticRegularizationCosts)
            criticRegularizationCosts = []
            if (settings['visualize_learning']):
                critic_regularization_viz.updateLoss(np.array(trainData["mean_critic_regularization_cost"]), np.array(trainData["std_critic_regularization_cost"]))
                critic_regularization_viz.redraw()
                critic_regularization_viz.setInteractiveOff()
                critic_regularization_viz.saveVisual(directory+"criticRegularizationGraph")
                critic_regularization_viz.setInteractive()
            
        if (settings['debug_actor']):
            
            mean_actorLosses = np.mean(actorLosses)
            std_actorLosses = np.std(actorLosses)
            trainData["mean_actor_loss"].append(mean_actorLosses)
            trainData["std_actor_loss"].append(std_actorLosses)
            actorLosses = []
            if (settings['visualize_learning']):
                actor_loss_viz.updateLoss(np.array(trainData["mean_actor_loss"]), np.array(trainData["std_actor_loss"]))
                actor_loss_viz.redraw()
                actor_loss_viz.setInteractiveOff()
                actor_loss_viz.saveVisual(directory+"actorLossGraph")
                actor_loss_viz.setInteractive()
            
            mean_actorRegularizationCosts = np.mean(actorRegularizationCosts)
            std_actorRegularizationCosts = np.std(actorRegularizationCosts)
            trainData["mean_actor_regularization_cost"].append(mean_actorRegularizationCosts)
            trainData["std_actor_regularization_cost"].append(std_actorRegularizationCosts)
            actorRegularizationCosts = []
            if (settings['visualize_learning']):
                actor_regularization_viz.updateLoss(np.array(trainData["mean_actor_regularization_cost"]), np.array(trainData["std_actor_regularization_cost"]))
                actor_regularization_viz.redraw()
                actor_regularization_viz.setInteractiveOff()
                actor_regularization_viz.saveVisual(directory+"actorRegularizationGraph")
                actor_regularization_viz.setInteractive()
    
        

if __name__ == '__main__':
    
    fitModelToData(sys.argv[1])
    