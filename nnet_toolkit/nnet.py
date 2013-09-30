import numpy as np
        
class layer(object):
    def __init__(self,node_count,activation='squash',step_size=None,dropout=None,select_func=None,select_func_params=None):
        self.node_count = node_count
        self.activation = activation
        self.step_size = step_size
        #tells the percentage of neurons to keep active
        self.dropout = dropout
        
        #parameters related to experimental research code. These can be ignored for normal use.
        self.select_func = select_func
        self.select_func_params=select_func_params
        self.selected_neurons = None;
        
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
            #the bottom row is the weights for the bias neuron
            # -- this neuron is set to 1.0 and these weights are essentially ignored
            l.weights = C*2*(np.random.random([l.node_count_output+1, l.node_count_input+1]) - 0.5)

    def zero_gradients(self):
        for l in self.layer:
            l.gradient = np.zeros(l.weights.shape)

    @property
    def input(self):
        self._input.copy_to_host()
        return self._input.numpy_array

    @input.setter
    def input(self,value):
        self._input = value
        self._input = np.append(self._input,np.ones((1,self._input.shape[1])),axis=0)
    
    @input.deleter
    def input(self):
        del self._input

    def feed_forward(self,input=None):
        #optionally allow passing input as an argument
        if input is not None:
            self.input = input

        #NOTE: a possible speedup here would be not to reconstruct the matrix, but to
        #fill it in each time.
        for index,l in enumerate(self.layer):
            if(index == 0):
                input = self._input
            else:
                input = self.layer[index-1].output
            l.input = input
            #print(str(index) + " " + str(l.weights.shape) + " " + str(l.input.shape))
            l.weighted_sums = np.dot(l.weights,l.input)
            
            #apply activation function
            if(l.activation == 'squash'):
                l.output = l.weighted_sums / (1+np.abs(l.weighted_sums))
            elif(l.activation == 'sigmoid'):
                l.output = 1/(1 + np.exp(-1*l.weighted_sums))
            elif(l.activation == 'tanh'):
                l.output = 1.7159*np.tanh((2.0/3.0)*l.weighted_sums)
                #TODO: softmax? others?
            elif(l.activation == 'linear_rectifier'):
                l.output = np.maximum(0,l.weighted_sums)
            else: #base case is linear
                l.output = l.weighted_sums
            
            if(l.select_func is not None):
                l.select_func(l,l.select_func_params);
                
            if(l.dropout is not None and self.train == True):
                if(l.activation == 'linear_rectifier'):
                    if(l.dropout == 0.5):
                        l.output = l.output*np.random.randint(0,2,l.output.shape);
                    else:
                        l.output = l.output*np.random.binomial(1,l.dropout,l.output.shape);
                else:
                    if(l.dropout == 0.5):
                        l.d_selected = np.random.randint(0,2,l.output.shape);
                        l.output = l.output*l.d_selected
                    else:
                        l.d_selected = np.random.binomial(1,l.dropout,l.output.shape);
                        l.output = l.output*l.d_selected
                        
            #TODO: Test this - it may be bugged! I think I should be doing: l.output = l.output*(l.dropout); 
            elif(l.dropout is not None and self.train == False):
                l.output = l.output*(1.0 - l.dropout);
                
            #one row in output is bias, set it to 1
            #note that bias is enabled even if dropout disabled it.
            l.output[-1,:] = 1.0


        #ignore last row for network output
        self.output = self.layer[len(self.layer)-1].output[0:-1,:]

    def back_propagate(self,error=None):
        if(error is not None):
            self.error = error

        #python doesn't easily allow reversed(enumerate()) - use this instead
        for l in reversed(self.layer):
            #if we're on the last layer
            #print(str(index));
            if(l.index == len(self.layer)-1):
                #must do this to account for the bias
                delta_temp = np.append(self.error,np.zeros((1,self.error.shape[1])),axis=0)
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
                #This stores them as bools - but it doesn't matter
                l.activation_derivative = np.greater(l.output,0);    
            else: #base case is linear
                l.activation_derivative = np.ones(l.output.shape);


                
            #bottom row of activation derivative is the bias 'neuron'
            #it's derivative is always 0
            l.activation_derivative[-1,:] = 0.0

            l.delta = l.activation_derivative*delta_temp;
            
            #zero out any deltas for neurons that were selected
            #note: selected_neurons means the neuron was selected for being
            #deactivated.
            if(l.selected_neurons is not None):
                l.delta[l.selected_neurons] = 0;
                l.selected_neurons = None;
            
            if(l.dropout is not None and self.train == True):
                if(l.activation != 'linear_rectifier'):
                    l.delta = l.delta*l.d_selected
            #calculate weight gradient
            l.gradient = l.gradient + np.dot(l.delta,l.input.transpose());
        self.epoch_size = self.epoch_size + self._input.shape[1];

    def update_weights(self):
        #Prevent calling update_weights() without calling back_propagate first
        #(with a non-empty vector) from crashing.
        if(self.epoch_size == 0):
            return;
            
        for l in reversed(self.layer):
            l.weight_change = -l.step_size*l.gradient/self.epoch_size;
            l.weights = l.weights + l.weight_change;
            l.gradient = np.zeros(l.weights.shape);
        self.epoch_size = 0;
