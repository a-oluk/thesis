from matplotlib import pyplot as plt
import numpy as np





# x = np.linspace(1,100,10)


# plt.figure()
# plt.plot(function(x))
# plt.show()

# Function f as input
# Around linspace
# how many points per point ,k
def noise_on_data(x, k=10):
    k = 10
    x_new = x
    for i in np.arange(0,k): #
        x_new = np.concatenate((x_new, x))

    x_random = np.random.normal(0,0.1,len(x_new))*15  # Rauschen generieren
    y_new = function(x_new) + x_random
    return x_new, y_new

def function(x):
    a = 1
    b = 1
    return (x - 3) ** 2 + 1


x = np.arange(0, 10, 1)
x1, y1 = noise_on_data(x)

mymodel = np.poly1d(np.polyfit(x1, y1, 3))
myline = np.linspace(0, 10, 100)
plt.figure()
plt.scatter(x1, y1)
plt.plot(myline, ((myline- 3) ** 2 + 1))
plt.plot(myline, mymodel(myline))

plt.figure(2)
diff = mymodel(myline)-((myline- 3) ** 2 + 1)

plt.plot(myline, diff)
plt.show()
