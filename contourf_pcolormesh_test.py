import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

x = np.array([-10,-5,-1,2,3,10,20,22,25,30,-5,-1,2,3,10,20,22,25,30,18,17,12])
y = np.array([-3,-3,-2,2,5,6,7,10,14,22,6,7,10,14,22,6,7,10,14,22,6,7])


z = (y-2)**2+ (x-4)**3 + np.random.normal(0,1,len(x))
f = interpolate.interp2d(x, y, z, kind = 'cubic')
Z2 = f(x, y)

fig = plt.figure(1)

plt.scatter(x,y,s=200,c= z, cmap= "viridis")
plt.colorbar()
d = {'X': x, 'Y': y}
df = pd.DataFrame(data=d)
final_df = df.sort_values(by=['X'], ascending=True)
xdata = df['X'].to_numpy()
ydata= df['Y'].to_numpy()

X2, Y2 = np.meshgrid(xdata, ydata)
#print(len(x),len(y),len(Z2))
fig = plt.figure(2)
plt.pcolormesh(xdata,ydata,Z2) #Create a pseudocolor plot with a non-regular rectangular grid.
plt.colorbar()

#
plt.figure(3)
plt.contourf(Z2)
#plt.axes([0 ,30,0,20])
plt.show()

