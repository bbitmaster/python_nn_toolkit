import numpy as np
		
class layer(object):
	def __init__(self,node_count,activation='squash'):
		self.node_count = node_count;
		self.activation = 'squash';
		self.step_size = .05;
		pass;
		
class net:
	def __init__(self,layer,parameters=None):
		self.re_init(layer,parameters);
		
	def re_init(self,layer,parameters):
		#don't store first layer since it is simply an input layer
		self.layer = layer[1:len(layer)];
		
		for i in range(len(self.layer)):
			self.layer[i].node_count_input  = layer[i].node_count;
			self.layer[i].node_count_output = layer[i+1].node_count;	
		
		#we may want to be able to quickly loop over the layer
		#and know the index
		#for i in range(len(self.layer)):
		#	self.layer[i].index = i;
		self.initialize_weights();
		self.zero_gradients();

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
			l.gradient = np.zeros(l.weights.shape);

	def feed_forward(self,input=None):
		#optionally allow passing input as an argument
		if input is not None:
			self.input = input

		#NOTE: a possible speedup here would be not to reconstruct the matrix, but to
		#fill it in each time.
		for index,l in enumerate(self.layer):
			if(index == 0):
				input = self.input;
			else:
				input = self.layer[index-1].output;

			l.input = np.append(input,np.ones((1,input.shape[1])),axis=0);
			#print(str(index) + " " + str(l.weights.shape) + " " + str(l.input.shape))
			l.weighted_sums = np.dot(l.weights,l.input);
			
			#apply activation function
			if(l.activation == 'squash'):
				l.output = l.weighted_sums / (1+np.abs(l.weighted_sums));
			elif(l.activation == 'sigmoid'):
				l.output = 1/(1 + np.exp(-1*l.weighted_sums));
				#TODO: linear rectified? softmax? others?
			else: #base case is linear
				l.output = l.weighted_sums
			#TODO: dropout?
		self.output = self.layer[len(self.layer)-1].output;

	def back_propagate(self,error=None):
		if(error is not None):
			self.error = error

		#python doesn't easily allow reversed(enumerate()) - use this instead
		index = len(self.layer)-1;
		for l in reversed(self.layer):

			#if we're on the last layer
			#print(str(index));
			if(index == len(self.layer)-1):
				delta_temp = self.error;
			else:
				#print(str(index) + " " + str(self.layer[index+1].weights.transpose().shape) + " " + str(self.layer[index+1].delta.shape))
				delta_temp = np.dot(self.layer[index+1].weights.transpose(),self.layer[index+1].delta);
			
			if(l.activation == 'squash'):
				l.activation_derivative = 1.0/((1+np.abs(l.weighted_sums)**2))
			elif(l.activation == 'sigmoid'):
				l.activation_derivative = l.output*(1 - l.output);
			else: #base case is linear
				l.activation_derivative = np.ones(l.output.shape);

			#add row to layer to account for bias
			if(index < len(self.layer)-1):
				l.activation_derivative = np.append(l.activation_derivative,np.zeros((1,l.activation_derivative.shape[1])),axis=0)
			l.delta = l.activation_derivative*delta_temp;
			
			#chop off bottom row
			if(index < len(self.layer)-1):
				l.delta = l.delta[0:len(l.delta)-1];

			#calculate weight gradient
			l.gradient = l.gradient + np.dot(l.delta,l.input.transpose());
			index = index-1

	def update_weights(self):
		index = len(self.layer)-1;
		for l in reversed(self.layer):
			l.weight_change = -l.step_size*l.gradient;
			l.weights = l.weights + l.weight_change;
			l.gradient = np.zeros(l.weights.shape);
			index = index - 1;

