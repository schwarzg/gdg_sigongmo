import numpy as np

#Generate sample data
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]]))

#layer function
def hidden(x,w,b,act):
	return act(w*x+b)

