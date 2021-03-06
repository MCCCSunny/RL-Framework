import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class DeepNNSingleNetDropout(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNSingleNetDropout,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        # create a small convolutional neural network
        # inputLayer = lasagne.layers.InputLayer((None, self._state_length), self._State)
        # new_state = theano.tensor.concatenate([self._State, self._Action], axis=1)
        # input = lasagne.layers.InputLayer((None, self._state_length + self._action_length), new_state)
        # stateInput = lasagne.layers.SliceLayer(input, indices=slice(0, self._state_length), axis=1)
        # actionInput = lasagne.layers.SliceLayer(input, indices=slice(self._state_length+1, self._state_length + self._action_length), axis=1)
        stateInput = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = stateInput.input_var
        actionInput = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = actionInput.input_var
        
        
        """
        network = lasagne.layers.DenseLayer(
                inputLayerA, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        network = lasagne.layers.DropoutLayer(stateInput, p=self._dropout_p, rescale=True)
        network = lasagne.layers.DenseLayer(
                network, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        networkMiddle = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                networkMiddle, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._actor = lasagne.layers.DenseLayer(
                network, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
        network = lasagne.layers.DenseLayer(
                networkMiddle, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        network = lasagne.layers.DenseLayer(
                network, num_units=16,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
    
        # print "Initial W " + str(self._w_o.get_value()) 

        # actionInput = lasagne.layers.InputLayer((None, self._action_length), self._Action)        
        networkMiddleFD = lasagne.layers.ConcatLayer([networkMiddle, actionInput], axis=1)
        
        network = lasagne.layers.DenseLayer(
                networkMiddleFD, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        self._forward_dynamics_net = lasagne.layers.DenseLayer(
                network, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
        network = lasagne.layers.DenseLayer(
                networkMiddleFD, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
