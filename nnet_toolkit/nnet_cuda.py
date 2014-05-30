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

from nnet import *
import numpy as np
import cudamat as cm
#from cudamat import learn as cl

class net_cuda(net):
	def __init__(self,layer,step_size=None,dropout=None):
		#TODO: should probably put cudamat initialization elsewhere
		#in case it is used by more than one network
		cm.cuda_set_device(0)
		cm.init()
		super(net_cuda,self).__init__(layer,step_size,dropout)

	def initialize_weights(self):
		super(net_cuda,self).initialize_weights()
		for index,l in enumerate(self.layer):
			l.weights = cm.CUDAMatrix(l.weights)

	def zero_gradients(self):
		#TODO: make empty matrix and set it to 0
		for l in self.layer:
			l.gradient = cm.CUDAMatrix(np.zeros(l.weights.shape))

	@property
	def input(self):
		return self._input

	@input.setter
	def input(self,value):
		self._input = value
		self._input = cm.CUDAMatrix(np.append(value,np.ones((1,value.shape[1])),axis=0))
	
	@input.deleter
	def input(self):
		del self._input


	def feed_forward(self,input=None):
		#optionally allow passing input as an argument
		if input is not None:
			self.input = input

		for index,l in enumerate(self.layer):
			if(index == 0):
				input = self.input
			else:
				input = self.layer[index-1].output

			l.input = input
			#print(str(index) + " " + str(l.weights.shape) + " " + str(l.input.shape))
			l.weighted_sums = cm.dot(l.weights,l.input)
			
			#apply activation function
			if(l.activation == 'squash'):
				pass
				#TODO: write kernal for this
				#l.output = l.weighted_sums / (1+np.abs(l.weighted_sums))
			elif(l.activation == 'sigmoid'):
				l.output = l.weighted_sums.apply_sigmoid()
			#elif(l.activation == 'linear_rectifier'):
			#	l.output = np.maximum(0,l.weighted_sums)
			else: #base case is linear
				l.output = l.weighted_sums
			#if(l.dropout is not None and self.train == True):
			#	if(l.dropout == 0.5):
			#		l.output = l.output*np.random.randint(0,2,l.output.shape);
			#	else:
			#		l.output = l.output*np.random.binomial(1,l.dropout,l.output.shape);
			#elif(l.dropout is not None and self.train == False):
			#	l.output = l.output*(1.0 - l.dropout);
		self.output = self.layer[len(self.layer)-1].output
		self.output.copy_to_host()
		self.output = self.output.numpy_array
		self.output = self.output[0:-1,:]

	def back_propagate(self,error=None):
		if(error is not None):
			self.error = cm.error

		#python doesn't easily allow reversed(enumerate()) - use this instead
		for l in reversed(self.layer):
			#if we're on the last layer
			#print(str(index));
			if(l.index == len(self.layer)-1):
				delta_temp = cm.CUDAMatrix(np.append(self.error,np.zeros((1,self.error.shape[1])),axis=0))
			else:
				#Possible TODO?: is there a way to get rid of this transpose? it is slow to have to do this
				#delta_temp = cm.empty((self.layer[l.index+1].weights.shape[1],self.layer[l.index+1].weights.shape[0]))
				#delta_temp = self.layer[l.index+1].weights.transpose()
				self.layer[l.index+1].weights.set_trans(True);
				delta_temp = cm.dot(self.layer[l.index+1].weights,self.layer[l.index+1].delta);
				self.layer[l.index+1].weights.set_trans(False);
			
			if(l.activation == 'squash'):
				pass
				#l.activation_derivative = 1.0/((1+np.abs(l.weighted_sums)**2))
			elif(l.activation == 'sigmoid'):
				#l.activation_derivative = cm.empty(l.output.shape);
				l.output.apply_logistic_deriv(l.output)
				l.activation_derivative = l.output
			#elif(l.activation == 'linear_rectifier'):
				#1 if greater than 0, 0 otherwise.
				#This stores them as bools - but it doesn't matter
				#l.activation_derivative = np.greater(l.output,0);	
			else: #base case is linear
				l.activation_derivative = cm.empty(l.output.shape);
				l.activation_derivitive.assign_scalar(1.0)
			
			#bottom row of activation derivative is the bias 'neuron'
			l.delta = cm.empty(delta_temp.shape)

			l.activation_derivative.mult(delta_temp,target=l.delta)
			
			#calculate weight gradient
			#input_t = cm.empty((l.input.shape[1],l.input.shape[0]))
			#input_t = l.input.transpose()
			l.input.set_trans(True)
			l.gradient.add_dot(l.delta,l.input);
			l.input.set_trans(False)
		self.epoch_size = self.epoch_size + self.input.shape[1];
	def update_weights(self):
		for l in reversed(self.layer):
			#l.weight_change = -l.step_size*l.gradient/self.epoch_size;
			l.gradient.mult(-l.step_size/self.epoch_size)
			l.weights.add(l.gradient);
			l.gradient.assign_scalar(0.0);
		self.epoch_size = 0;
