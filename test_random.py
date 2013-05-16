import numpy as np;

import nnet_toolkit;

layers = [nnet_toolkit.layer(2),nnet_toolkit.layer(32),nnet_toolkit.layer(1)];

training_data = np.array([[0,0,1,1],[0,1,0,1]]);
training_out = np.array([0,1,1,0]);

net = nnet_toolkit.net(layers);

net.input = training_data;
for i in range(1000000):
	net.feed_forward();
	net.error = net.output - training_out;
	net.back_propagate();
	net.update_weights();
	if(i%1000 == 0):
		print(str(net.error));

