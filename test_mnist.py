import time
import numpy as np
import data.mnist as mnist
from nnet_toolkit import nnet


# Load Data
features,labels = mnist.read(range(9),dataset='training')
tfeatures,tlabels = mnist.read(range(9),dataset='testing')

# Initialize Network
layers = [nnet.layer(features.shape[1]),nnet.layer(128,'sigmoid'),nnet.layer(labels.shape[1],'sigmoid')];
net = nnet.net(layers,step_size=.1);


# Train Network for N epochs
N = 50
mini_batch_size = 1000
t = time.time();
print "Starting Training..."
for epoch in range(N):
    # Randomize Features
    rix = np.random.permutation(features.shape[0])
    features = features[rix]
    labels = labels[rix]
    net.input = features;
    
    # Training
    for i in range(0,features.shape[0],mini_batch_size):
        net.input = features[i:i+mini_batch_size].T
        net.feed_forward();
        net.error = net.output - labels[i:i+mini_batch_size].T;
        net.back_propagate();
        net.update_weights();
    
    # Testing
    ncorrect = 0.0
    for i in range(0,tfeatures.shape[0],mini_batch_size):
        net.input = tfeatures[i:i+mini_batch_size].T
        net.feed_forward();
        ncorrect += np.sum(np.argmax(net.output,axis=0)==
                                 np.argmax(tlabels[i:i+mini_batch_size].T,axis=0))
    
    # Print Statistics
    t_delta = time.time() - t;
    print("Epoch: " + str(epoch) + " took " + str(t_delta) + " seconds to complete")
    print("Testing Accuracy: " + str(ncorrect/float(tfeatures.shape[0])))
    t = time.time();

