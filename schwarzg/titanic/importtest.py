import sys
print sys.path

sys.path.append('/home/schwarzg/Source_code/Machinelearning/RNN/schwarzg')
print sys.path

import J_ml.activation as act

import numpy as np
A=np.arange(10)-5
B=act.relu(A)
print B
