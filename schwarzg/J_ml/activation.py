import numpy as np

def sig(z):
	return 1.0/(1.0+np.exp(-z))
	
def sigd(z):
	sigarr=sig(z)
	return np.multiply(sigarr,(1.0-sigarr))

def relu(z):
	return np.maximum(0,z)
	
def relud(z):
	return np.piecewise(z,[z<0,z>=0],[0,1]) 
	
def lrelu(z):
	return np.maximum(0.1*z,z)
	
def lrelud(z):
	return np.piecewise(z,[z<0,z>=0],[0.1,1]) 

