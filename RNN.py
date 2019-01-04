import numpy as np

def toy_problem(T=100.,amp=0.05):
	x=np.arange(0,2*T+1)
	noise=amp*np.random.uniform(low=-1,high=1,size=len(x))
	return np.sin(2.0*np.pi*x/T)+noise


