#Author: Ben Goodrich
#This is experimental research code that is used to "select" neurons in the hidden layer based on some criteria

import numpy as np

#select based on minimum distance to dot product=0 hyperplane. 
#this version does not normalize to get actual euclidean distance
def minabs_select_func(self,params):
    n_active_count = params;
    activation_abs = np.abs(self.weighted_sums);
    #place smallest activations in top rows
    sorted_activations = np.sort(activation_abs,axis=0)
    #select the n'th smallest activation, and set everything >= it to 0
    self.selected_neurons = activation_abs >= sorted_activations[n_active_count,:]
    self.output[self.selected_neurons] = 0;

#select based on maximum distance to dot product=0 hyperplane. 
#this version does not normalize to get actual euclidean distance
def maxabs_select_func(self,params):
    n_active_count = params;
    activation_abs = np.abs(self.weighted_sums);
    #place smallest activations in top rows
    sorted_activations = np.sort(activation_abs,axis=0)
    #select the n'th largest activation (-n_active_count means nth largest)
    #set everything less than it to 0
    self.selected_neurons = activation_abs < sorted_activations[-n_active_count,:]
    self.output[self.selected_neurons] = 0;


#select based on most negative activation. (selects the smallest activation where smallest)
#is defined as the lowest real value)
def most_negative_select_func(self,params):
    n_active_count = params;
    activation = self.weighted_sums
    #place smallest activations in top rows
    sorted_activations = np.sort(activation,axis=0)
    #select the n'th smallest activation, and set everything >= it to 0
    self.selected_neurons = activation >= sorted_activations[n_active_count,:]
    self.output[self.selected_neurons] = 0;

#select based on minimum euclidean distance to dot product=0 hyperplane. 
def minabs_select_func_normalized(self,params):
    n_active_count = params;
    #get the norm of each neurons weights.
    weight_norm = np.sum(self.weights**2,axis=1)**(1./2.);

    #need to do a row-wise divide (the transposes handle that)
    #each activation needs to be divided by the norm of that neuron's weights    
    normalized_activations = (self.weighted_sums.T/weight_norm.T).T
    activation_abs = np.abs(normalized_activations);
    #place smallest activations in top rows
    sorted_activations = np.sort(activation_abs,axis=0)
    #select the n'th smallest activation, and set everything >= it to 0
    self.selected_neurons = activation_abs >= sorted_activations[n_active_count,:]
    self.output[self.selected_neurons] = 0;


#select based on maximum euclidean distance to dot product=0 hyperplane. 
def maxabs_select_func_normalized(self,params):
    n_active_count = params;
    #get the norm of each neurons weights.
    weight_norm = np.sum(self.weights**2,axis=1)**(1./2.);

    #need to do a row-wise divide (the transposes handle that)
    #each activation needs to be divided by the norm of that neuron's weights    
    normalized_activations = (self.weighted_sums.T/weight_norm.T).T
    activation_abs = np.abs(normalized_activations);
    #place smallest activations in top rows
    sorted_activations = np.sort(activation_abs,axis=0)
    #select the n'th largest activation (-n_active_count means nth largest)
    #set everything less than it to 0
    self.selected_neurons = activation_abs < sorted_activations[-n_active_count,:]
    self.output[self.selected_neurons] = 0;


#select based on most negative activation. (selects the smallest activation where smallest)
def most_negative_select_func_normalized(self,params):
    n_active_count = params;
    #get the norm of each neurons weights.
    weight_norm = np.sum(self.weights**2,axis=1)**(1./2.);

    #need to do a row-wise divide (the transposes handle that)
    #each activation needs to be divided by the norm of that neuron's weights    
    normalized_activations = (self.weighted_sums.T/weight_norm.T).T
    activation = normalized_activations
    #place smallest activations in top rows
    sorted_activations = np.sort(activation,axis=0)
    #select the n'th smallest activation, and set everything >= it to 0
    self.selected_neurons = activation >= sorted_activations[n_active_count,:]
    self.output[self.selected_neurons] = 0;