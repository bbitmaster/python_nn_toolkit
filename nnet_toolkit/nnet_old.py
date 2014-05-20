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
		
class layer(object):
	def __init__(self,node_count,activation='squash',step_size=None,dropout=None):
		self.node_count = node_count
		self.activation = activation
		self.step_size = step_size
		self.dropout = dropout
		pass;
		
class net(object):
	def __init__(self,layer,step_size=None,dropout=None):
		#don't store first layer since it is simply an input layer
		self.layer = layer[1:len(layer)]

		
		for i in range(len(self.layer)):
			self.layer[i].node_count_input  = layer[i].node_count
			self.layer[i].node_count_output = layer[i+1].node_count	
		
		#we may want to be able to quickly loop over the layer
		#and know the index
		for i in range(len(self.layer)):
			self.layer[i].index = i

		for l in self.layer:
			if(step_size is not None):
				l.step_size = step_size;
			if(dropout is not None):
				l.dropout = dropout
		self.layer[len(self.layer)-1].dropout = None
		self.initialize_weights()
		self.zero_gradients()
		self.epoch_size = 0
		self.train = True
	def initialize_weights(self):
		for index,l in enumerate(self.layer):
			if index == 0:
				C = 1.3/np.sqrt(1 + (l.node_count_input+1)*0.5 )
			else:
				C = 1.3/np.sqrt(1 + (l.node_count_input+1)*0.3 )
			l.weights = C*2*(np.random.random([l.node_count_output, l.node_count_input+1]) - 0.5)
			#print("l.weights.shape: " + str(l.weights.shape));

	def zero_gradients(self):
		for l in self.layer:
			l.gradient = np.zeros(l.weights.shape)

	def feed_forward(self,input=None):
		#optionally allow passing input as an argument
		if input is not None:
			self.input = input

		#NOTE: a possible speedup here would be not to reconstruct the matrix, but to
		#fill it in each time.
		for index,l in enumerate(self.layer):
			if(index == 0):
				input = self.input
			else:
				input = self.layer[index-1].output

			l.input = np.append(input,np.ones((1,input.shape[1])),axis=0)
			#print(str(index) + " " + str(l.weights.shape) + " " + str(l.input.shape))
			l.weighted_sums = np.dot(l.weights,l.input)
			
			#apply activation function
			if(l.activation == 'squash'):
				l.output = l.weighted_sums / (1+np.abs(l.weighted_sums))
			elif(l.activation == 'sigmoid'):
				l.output = 1/(1 + np.exp(-1*l.weighted_sums))
				#TODO: linear rectified? softmax? others?
			elif(l.activation == 'linear_rectifier'):
				l.output = np.maximum(0,l.weighted_sums)
			else: #base case is linear
				l.output = l.weighted_sums
			if(l.dropout is not None and self.train == True):
				if(l.dropout == 0.5):
					l.output = l.output*np.random.randint(0,2,l.output.shape);
				else:
					l.output = l.output*np.random.binomial(1,l.dropout,l.output.shape);
			elif(l.dropout is not None and self.train == False):
				l.output = l.output*(1.0 - l.dropout);
		self.output = self.layer[len(self.layer)-1].output

	def back_propagate(self,error=None):
		if(error is not None):
			self.error = error

		#python doesn't easily allow reversed(enumerate()) - use this instead
		for l in reversed(self.layer):
			#if we're on the last layer
			#print(str(index));
			if(l.index == len(self.layer)-1):
				delta_temp = self.error;
			else:
				#print(str(index) + " " + str(self.layer[index+1].weights.transpose().shape) + " " + str(self.layer[index+1].delta.shape))
				delta_temp = np.dot(self.layer[l.index+1].weights.transpose(),self.layer[l.index+1].delta);
			
			if(l.activation == 'squash'):
				l.activation_derivative = 1.0/((1+np.abs(l.weighted_sums)**2))
			elif(l.activation == 'sigmoid'):
				l.activation_derivative = l.output*(1 - l.output);
			elif(l.activation == 'linear_rectifier'):
				#1 if greater than 0, 0 otherwise.
				#This stores them as bools - but it doesn't matter
				l.activation_derivative = np.greater(l.output,0);	
			else: #base case is linear
				l.activation_derivative = np.ones(l.output.shape);

			#add row to layer to account for bias
			if(l.index < len(self.layer)-1):
				l.activation_derivative = np.append(l.activation_derivative,np.zeros((1,l.activation_derivative.shape[1])),axis=0)
			l.delta = l.activation_derivative*delta_temp;
			
			#chop off bottom row
			if(l.index < len(self.layer)-1):
				l.delta = l.delta[0:len(l.delta)-1];

			#calculate weight gradient
			l.gradient = l.gradient + np.dot(l.delta,l.input.transpose());
		self.epoch_size = self.epoch_size + self.input.shape[1];
	def update_weights(self):
		for l in reversed(self.layer):
			l.weight_change = -l.step_size*l.gradient/self.epoch_size;
			l.weights = l.weights + l.weight_change;
			l.gradient = np.zeros(l.weights.shape);
		self.epoch_size = 0;
