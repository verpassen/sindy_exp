import numpy as np 
import pysindy as ps 

t = np.linspace(0,1,100)
x = 3*np.exp(-2*t)
y = 5*np.exp(t)

x = np.stack((x,y),axis=-1)
print(x)

model = ps.SINDy(feature_names=['x','y'])
model.fit(x,t=t)
model.print()
