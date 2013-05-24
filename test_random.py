import time
import numpy as np;

from nnet_toolkit import nnet;

#layers = [nnet_toolkit.layer(2),nnet_toolkit.layer(128,'squash'),nnet_toolkit.layer(1,'squash')];
layers = [nnet.layer(400),nnet.layer(128,'squash'),nnet.layer(1,'squash')];

#training_data = np.array([[0,0,1,1],[0,1,0,1]]);
#training_out = np.array([0,1,1,0]);

training_data = np.random.random((400,500));
training_out = np.random.random((1,500));

net = nnet.net(layers,step_size=.1);

net.input = training_data;
t = time.time();
for i in range(100000):
	net.feed_forward();
	net.error = net.output - training_out;
	net.back_propagate();
	net.update_weights();
	if(i%1000 == 0):
		#print("iteration: " + str(i) + " " + str(net.error));
		t_delta = time.time() - t;
		print("iteration: " + str(i) + " " + str(t_delta))
		t = time.time();

