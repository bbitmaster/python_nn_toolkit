import time
import numpy as np;

#from nnet_toolkit import nnet_old as nnet
from nnet_toolkit import nnet

layers = [nnet.layer(2),nnet.layer(128,'squash'),nnet.layer(1,'squash')];
#layers = [nnet_toolkit.layer(2),nnet_toolkit.layer(256,'linear_rectifier'),nnet_toolkit.layer(128,'linear_rectifier'),nnet_toolkit.layer(64,'linear_rectifier'),nnet_toolkit.layer(32,'linear_rectifier'),nnet_toolkit.layer(1,'squash')];

training_data = np.array([[0,0,1,1],[0,1,0,1]]);
training_out = np.array([0,1,1,0]);

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
		print("iteration: " + str(i) + " " + str(np.sum(net.error**2)) + ' ' + str(t_delta));
		t = time.time();
