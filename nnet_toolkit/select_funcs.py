#Author: Ben Goodrich

import numpy as np

#select based on maximum distance to dot product=0 hyperplane. 
#this version does not normalize to get actual euclidean distance
def lwta_select_func(self):
    nodes_per_group = self.nodes_per_group
    num_groups = self.weighted_sums.shape[0]/nodes_per_group
    batch_size = self.weighted_sums.shape[1]
    #print('weighted_sums shape: ' + str(self.weighted_sums.shape))
    #print('num_groups: ' + str(num_groups) + ' nodes_per_group: ' + str(nodes_per_group) + ' batch_size: ' + str(batch_size))
    
    #note we remove the bias node for this grouping
    activations_grouped = np.reshape(self.weighted_sums[0:-1,:],(num_groups,nodes_per_group,batch_size))

    #need to swap axes for the broadcasting to work properly
    activations_grouped = np.swapaxes(activations_grouped,0,1);
    
    #we want "True" neurons that are selected FOR REMOVAL
    activations_selected = (activations_grouped != np.max(activations_grouped,axis=0))
    #swap axes back and reshape back to 2d matrix
    activations_selected = np.swapaxes(activations_selected,0,1)
    activations_selected = np.reshape(activations_selected,(num_groups*nodes_per_group,batch_size))

    #append the bias back
    #print('selected neurons dtype: ' + str(activations_selected.dtype))
    self.selected_neurons = np.append(activations_selected,np.ones((1,activations_selected.shape[1]),dtype=np.bool),axis=0)
    #print('selected neurons dtype: ' + str(self.selected_neurons.dtype))
    self.output[self.selected_neurons] = 0;

def maxout_select_func(self):
    nodes_per_group = self.nodes_per_group
    num_groups = self.weighted_sums.shape[0]/nodes_per_group
    batch_size = self.weighted_sums.shape[1]
    #print('weighted_sums shape: ' + str(self.weighted_sums.shape))
    #print('num_groups: ' + str(num_groups) + ' nodes_per_group: ' + str(nodes_per_group) + ' batch_size: ' + str(batch_size))
    
    #note we remove the bias node for this grouping
    activations_grouped = np.reshape(self.weighted_sums[0:-1,:],(num_groups,nodes_per_group,batch_size))
    
    #need to swap axes for the broadcasting to work properly
    activations_grouped = np.swapaxes(activations_grouped,0,1);
    
    #we want "True" neurons that are selected FOR REMOVAL (so that their gradients will be passed back correctly)
    activations_selected = (activations_grouped != np.max(activations_grouped,axis=0))

    #swap axes back and reshape back to 2d matrix
    activations_selected = np.swapaxes(activations_selected,0,1)
    activations_selected = np.reshape(activations_selected,(num_groups*nodes_per_group,batch_size))
    
    #append the bias back. This matrix is used to correctly backpropagate through the gradients for weights that were selected.
    self.selected_neurons = np.append(activations_selected,np.ones((1,activations_selected.shape[1]),dtype=np.bool),axis=0)

    #for maxout we need to change the layer output so that only "groups" of neurons are output
    activations_grouped = np.max(activations_grouped,axis=0) #NOTE: axes 0 and 1 have been swapped here
    activations_grouped = np.append(activations_grouped,np.ones((1,activations_grouped.shape[1]),dtype=self.weights.dtype),axis=0)
    self.output = activations_grouped
