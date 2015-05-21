#Copyright (c) 2014, Ben Goodrich
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.#
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
#THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import select_funcs as sf

class layer(object):
    def __init__(self,node_count,activation='squash',step_size=None,dropout=None,
                 momentum=None,maxnorm=None,use_float32=False,
                 select_func=None,initialization_scheme=None,nodes_per_group=None,
                 initialization_constant=None,sparse_penalty=None,sparse_target=None,
                 rms_prop_rate=None):
        self.node_count = node_count
        self.activation = activation
        self.step_size = step_size
        
        #tells the percentage of neurons to keep active
        self.dropout = dropout

        self.maxnorm = maxnorm
        self.momentum = momentum
        self.rms_prop_rate = rms_prop_rate

        #function used to select neurons
        #used for local winner take all, maxout, or your own selection function (k sparse autoencoders?)
        if(activation == 'lwta'):
            self.select_func = sf.lwta_select_func
        elif(activation == 'maxout'):
            self.select_func = sf.maxout_select_func
        else:
            self.select_func = select_func
        
        self.selected_neurons = None;
        self.nodes_per_group = nodes_per_group

        #parameters related to the weight initilization scheme
        self.initialization_scheme = initialization_scheme;
        self.initialization_constant = initialization_constant;
        
        #parameters related to sparse auto-encoder based on KL-divergence
        self.sparse_penalty = sparse_penalty
        self.sparse_target = sparse_target
        self.mean_estimate_count = None
        
        self.use_float32 = use_float32
        pass;
        
class net(object):
    def __init__(self,layer,step_size=None,dropout=None):

        #set up input and output node counts
        
        #NOTE: a maxout layer can have more outputs than there are inputs on the layer after
        #This is due to the grouping of nodes before they are passed to the next layer
        for i in range(1,len(layer)):
            #If previous layer was maxout, then nodes must be grouped to get inputs for this layer.
            if(layer[i-1].activation == 'maxout'):
                layer[i].node_count_input  = layer[i-1].node_count/layer[i-1].nodes_per_group
            else:
                layer[i].node_count_input  = layer[i-1].node_count
            layer[i].node_count_output = layer[i].node_count    
        
        #Store layers, but don't store first layer since it is simply the input layer
        self.layer = layer[1:len(layer)]
        
        #we may want to be able to quickly loop over the layer
        #and know the index
        for i in range(len(self.layer)):
            self.layer[i].index = i

        for l in self.layer:
            if(step_size is not None and l.step_size is None):
                l.step_size = step_size;
            if(dropout is not None):
                l.dropout = dropout
        self.layer[len(self.layer)-1].dropout = None
        self.initialize_weights()
        self.zero_gradients()
        
        #init momentum, and rmsprop
        for l in self.layer:
            if(l.momentum is not None):
                l.vel = np.zeros(l.weights.shape,dtype=l.weights.dtype)
            if(l.rms_prop_rate is not None):
                l.mean_square_avg = np.ones(l.weights.shape,dtype=l.weights.dtype)

            
        self.epoch_size = 0
        self.train = True

    def initialize_weights(self):
        for index,l in enumerate(self.layer):
            if(l.initialization_scheme == 'krizhevsky'):
                #taken from
                #'ImageNet Classification with Deep Convolutional Neural Networks'
                #Hinton et all
                l.weights = np.random.normal(0.0,.01,[l.node_count_output+1, l.node_count_input+1])
                l.weights[:,-1] = 1.0
            elif(l.initialization_scheme == 'glorot'):
                #taken from
                #'Understanding the difficulty of training deep feedforward neural networks'
                #Xavier Glorot, Yoshua Bengio
                C = np.sqrt(6)/np.sqrt(l.node_count_output + l.node_count_input + 1)
                if(l.initialization_constant is not None):
                    C = C*l.initialization_constant
                l.weights = C*2*(np.random.random([l.node_count_output+1, l.node_count_input+1]) - 0.5)
                #a large bias weight for LWTA and Maxout can make a unit win the max too often
                #as described in "An Emperical Investigation of Catastrophic Forgetting in Neural Networks"
                #We set bias weights to 0 for these types of nets
                if(l.activation == 'lwta' or l.activation == 'maxout'):
                    l.weights[:,-1] = 0.0
            elif(l.initialization_scheme == 'scawi'):
                #taken from
                #Statistically Controlled Weight Initialization (SCAWI)
                #Gian Paolo Drago and Sandro Ridella
                #there is a slight modification to the formula used for the
                #first layer
                if index == 0:
                    C = 1.3/np.sqrt(1 + (l.node_count_input+1)*0.5 )
                else:
                    C = 1.3/np.sqrt(1 + (l.node_count_input+1)*0.3 )
                #the bottom row is the weights for the bias neuron
                # -- this neuron output is always set to 1.0 and these weights are essentially ignored
                l.weights = C*2*(np.random.random([l.node_count_output+1, l.node_count_input+1]) - 0.5)
                if(l.activation == 'lwta' or l.activation == 'maxout'):
                    l.weights[:,-1] = 0.0
            elif(l.initialization_scheme == 'prelu'):
                #taken from "Delving Deep into Rectifiers: Surpassing Human-Level Performance on
                #ImageNet Classification"
                if(index == 0):
                    std = np.sqrt(1./l.node_count_input)
                else:
                    std = np.sqrt(2./l.node_count_input)
                l.weights = np.random.normal(0.0,std,[l.node_count_output+1, l.node_count_input+1])
                l.weights[:,-1] = 0.0
            else:
            #by default, use the method promoted in lecun's Effecient Backprop paper
                C = 1/np.sqrt(l.node_count_input+1)
                l.weights = C*2*(np.random.random([l.node_count_output+1, l.node_count_input+1]) - 0.5)
                #a large bias weight for LWTA and Maxout can make a unit win the max too often
                #as described in "An Emperical Investigation of Catastrophic Forgetting in Neural Networks"
                #We set bias weights to 0 for these types of nets
                if(l.activation == 'lwta' or l.activation == 'maxout'):
                    l.weights[:,-1] = 0.0

            if(l.use_float32):
                l.weights = np.asarray(l.weights,np.float32)

    def zero_gradients(self):
        for l in self.layer:
            l.gradient = np.zeros(l.weights.shape,dtype=l.weights.dtype)

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self,value):
        self._input = value
        self._input = np.append(self._input,np.ones((1,self._input.shape[1]),dtype=value.dtype),axis=0)
    
    @input.deleter
    def input(self):
        del self._input

    def feed_forward(self,input=None):
        #optionally allow passing input as an argument
        if input is not None:
            self.input = input

        for index,l in enumerate(self.layer):
            if(index == 0):
                input = self._input
            else:
                input = self.layer[index-1].output
            l.input = input
            l.weighted_sums = np.dot(l.weights,l.input)
            
            #apply activation function
            if(l.activation == 'squash'):
                l.output = l.weighted_sums / (1+np.abs(l.weighted_sums))
            elif(l.activation == 'sigmoid'):
                l.output = 1/(1 + np.exp(-1*l.weighted_sums))
            elif(l.activation == 'tanh'):
                l.output = 1.7159*np.tanh((2.0/3.0)*l.weighted_sums)
            elif(l.activation == 'linear_rectifier'):
                l.output = np.maximum(0,l.weighted_sums)
            elif(l.activation == 'softmax'):
                l.output = np.exp(l.weighted_sums)
                #ignore bottom row in the summation since it does not represent any class at all
                l.output = l.output/np.sum(l.output[0:-1,:],axis=0)
            else: #base case is linear
                l.output = l.weighted_sums
                
            if(l.sparse_penalty is not None):
                #first pass - compute the mean
                #every other pass - maintain moving average
                if(l.mean_estimate_count is None):
                    l.mean_estimate = np.mean(l.output,axis=1)
                    l.mean_estimate_count = 0
                else:
                    l.mean_estimate = 0.99*l.mean_estimate + .01*np.mean(l.output,axis=1);
                    l.mean_estimate_count = l.mean_estimate_count + 1;
            
            if(l.select_func is not None):
                l.select_func(l);
                
            if(l.dropout is not None and self.train == True):
                #Multiple code paths to optimize for speed. The common case for dropout is to use
                #with rectified linear activations. In that case we do not need to save
                #l.d_selected. l.d_selected is saved to allow gradients to be ignored for weights
                #that were dropped out. linear rectified gradients are ignored anyway (if the
                #output is 0).
                if(l.activation == 'linear_rectifier'):
                    if(l.dropout == 0.5): #randint is slightly faster, and 0.5 is a common case
                        l.output = l.output*(np.random.randint(0,2,l.output.shape).astype(np.float32));
                    else:
                        l.output = l.output*(np.random.binomial(1,(1 - l.dropout),l.output.shape).astype(np.float32));
                else:
                    if(l.dropout == 0.5):
                        l.d_selected = np.random.randint(0,2,l.output.shape).astype(np.float32);
                        l.output = l.output*l.d_selected
                    else:
                        l.d_selected = np.random.binomial(1,(1 - l.dropout),l.output.shape).astype(np.float32);
                        l.output = l.output*l.d_selected
                        
            elif(l.dropout is not None and self.train == False):
                l.output = l.output*(1.0 - l.dropout);
                
            #one row in output is bias, set it to 1
            #note that bias 'input' is enabled even if dropout disabled it.
            l.output[-1,:] = 1.0


        #ignore last row for network output
        self.output = self.layer[len(self.layer)-1].output[0:-1,:]

    def back_propagate(self,error=None):
        if(error is not None):
            self.error = error

        for l in reversed(self.layer):
            #if this is the last layer
            if(l.index == len(self.layer)-1):
                #must do this to account for the bias
                delta_temp = np.append(self.error,np.zeros((1,self.error.shape[1]),dtype=self.error.dtype),axis=0)
            else:
                delta_temp = np.dot(self.layer[l.index+1].weights.transpose(),self.layer[l.index+1].delta);
            if(l.activation == 'squash'):
                l.activation_derivative = 1.0/((1+np.abs(l.weighted_sums)**2))
            elif(l.activation == 'sigmoid'):
                l.activation_derivative = l.output*(1 - l.output);
            elif(l.activation == 'tanh'):
                #l.activation_derivative = ((2.0/3.0)/1.7159)*(1.7159**2 - l.output**2)
                l.activation_derivative = 0.3885230297025856*(2.94431281 - l.output*l.output)
            elif(l.activation == 'linear_rectifier'):
                #1 if greater than 0, 0 otherwise.
                #This stores as bools
                l.activation_derivative = np.greater(l.output,0);    
            else: #base case is linear or softmax (also applies to lwta and maxout)
                l.activation_derivative = np.ones(l.output.shape,dtype=l.output.dtype);

            #bottom row of activation derivative is the bias 'neuron'
            #it's derivative is always 0
            l.activation_derivative[-1,:] = 0.0
            
            #add sparsity error to delta
            #from http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
            if(l.sparse_penalty is not None):
                sparse_error = l.sparse_penalty*(-l.sparse_target/l.mean_estimate + (1.0 - l.sparse_target)/(1.0 - l.mean_estimate))
                delta_temp = delta_temp +  sparse_error[:,np.newaxis]
            
            l.delta = l.activation_derivative*delta_temp;

            #ignore deltas for weights that were dropped out.
            if(l.dropout is not None and self.train == True):
            #This is simply an optimization for speed. If we have rectified linear activations
            #then dropout makes the activation be 0. gradients should be ignored anyway for
            #0 activations. That gets rid of the need to do the below multiply.
                if(l.activation != 'linear_rectifier'):
                    l.delta = l.delta*l.d_selected

            #For maxout networks, we have a smaller weight matrix which means delta will be smaller than it should be
            #It must be enlarged here (via np.repeat). The path that the gradient takes is accounted for in l.selected_neurons
            #Bias neuron is removed then reinserted via np.append
            if(l.activation == 'maxout'):
                l.delta = np.repeat(l.delta[0:-1],l.nodes_per_group,axis=0)
                l.delta = np.append(l.delta,np.ones((1,l.delta.shape[1]),l.weights.dtype),axis=0)
            
            #zero out any deltas for neurons that were selected
            #note: "selected" means the neuron was selected for being deactivated.
            if(l.selected_neurons is not None):
                l.delta[l.selected_neurons] = 0;
                l.selected_neurons = None;
            
            #calculate weight gradient
            l.gradient = l.gradient + np.dot(l.delta,l.input.transpose());

        self.epoch_size = self.epoch_size + self._input.shape[1];

    def update_weights(self):
        #Prevent calling update_weights() without calling back_propagate first
        #(with a non-empty vector) from crashing.
        if(self.epoch_size == 0):
            return;
        for l in reversed(self.layer):
            if(l.rms_prop_rate is not None):
                l.mean_square_avg = l.rms_prop_rate*l.mean_square_avg + (1.0 - l.rms_prop_rate)*(l.gradient**2)
                l.gradient = l.gradient/(np.sqrt(l.mean_square_avg))
            l.weight_change = -l.step_size*l.gradient/self.epoch_size;
            if(l.momentum is not None):
                l.vel = l.momentum*l.vel + l.weight_change
                l.weight_change = l.vel
            l.weights = l.weights + l.weight_change;
            if(l.maxnorm is not None):
                weight_norm = np.sum(l.weights**2,axis=0)**0.5
                condition = weight_norm > l.maxnorm
                l.weights = l.maxnorm*(l.weights/weight_norm)*condition + l.weights*(1 - condition)
                
            l.gradient = np.zeros(l.weights.shape,dtype=l.weights.dtype);
        self.epoch_size = 0;
