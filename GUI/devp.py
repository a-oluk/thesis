
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
#import functions

#%%

# Just the test function
#f = lambda x: np.sin(0.9*x)*(x-2)
f_0 = lambda x: np.sin(5*x)*x+(x-2)

def f(x):
    if x<-1: 
        return np.sin(5*x)*x 
    elif x>=-1 and x<3:
        return (x-2)
    else:
        return np.sin(2*x)+x
#f = lambda x: np.sin(6*x)*0.4(x-2)

#%%

#### HIER KOMMT GAUS PROZESS REIN JETZT

'''Was gehört zu einem Gauss Prozess ? 
    -> 1.) Kernel-Function
    -> 2.) Gaus-Verteilung
            - Erwartungswert (mean)
            - Kovarianzfunktion (covariance) 
                - radial basis function
                - matern
                - kombination of 
    -> 3.) Neue Sampling Punkte bestimmen und Gaus Process neu anwenden
                
    Schritte: 
        1.) A priori Erwartungswertfunktion
        2.) A-priori Kovarianzfunktion
        3.) Feinabstimmung der parameter
        4.) Bedingte Verteilung
        5.) Interpretation
'''

#%% Not in Usage now
 
# not for my program yet
# from https://jcmwave.com/company/blog/item/1050-gaussian-process-regression
if False:
    X1= ...
    X2= ...

    def get_covariance_martices():
    #Covariance matrices
        Sigma11 = np.array([[rbf_kernel(x,y) for x in X1] for y in X1])
        Sigma12 = np.array([[rbf_kernel(x,y) for x in X2] for y in X1])
        Sigma21 = Sigma12.T
        Sigma22 = np.array([[rbf_kernel(x,y) for x in X2] for y in X2])
        return None
    
#%%  Kernel Function 

def exponential_cov(x, y, params):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

def rbf_kernel(X, Y, var=1, gamma=1.0):
    """Radial basis function kernel"""
    # Calculate the squared Euclidean distance between all pairs of points
    sq_dists = np.sum(X*2, 1).reshape(-1, 1) + np.sum(Y*2, 1) - 2 * np.dot(X, Y.T)
    return var * np.exp(-gamma * sq_dists)

#%% Conditional Function  WARUM BENTUTZE ICH DAS NICHT ?!?!?!
def conditional(x_new, x, y, params):
    #B = exponential_cov(x_new, x, params)     # Kernel of oversavtion vs to-predict
    #C = exponential_cov(x, x, params)         # Kernel of observations
    #A = exponential_cov(x_new, x_new, params) # Kernel of to-predict

    B = rbf_kernel(x_new, x)      # Kernel of oversavtion vs to-predict
    C = rbf_kernel(x, x)          # Kernel of observations
    A = rbf_kernel(x_new, x_new)  # Kernel of to-predict

    mu = np.linalg.inv(C).dot(B.T).T.dot(y)       # mean
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))  # covariance
    return (mu.squeeze(), sigma.squeeze())

#%% Prediction 
def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)                                # inverse of matrix -> use 2 times
    y_pred = np.dot(k, Sinv).dot(t)                            # k * sigma^-1 * t
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)  # COV_xx - K*sigma^-1*k
    return y_pred, sigma_new

###################################################################################################
#%%
N = 10
N_0 = 5
θ = [2, 10]    # parameters for COV
                              
σ_0 = exponential_cov(0, 0, θ)               # Cov = sigma = kernel

xpts = np.arange(-3, 3, step=0.01)           # arange ---> FOR WHAT ?
f_real_func = np.array([f(i) for i in xpts])
plt.figure()
plt.errorbar(xpts, np.zeros(len(xpts)) , yerr=σ_0) # Errorbar ; yerr = error size, capsize = 
plt.axis([-3,3,-3,3])
#plt.savefig('first.png')

#%% Generate RANDOM DATASET
np.random.seed(10)
x = np.random.uniform(-3,3,N_0).tolist()
#x= [-2,0,2] 
y= np.array([f(i) for i in x]).tolist() # FOR INITIAL DATASET
#y = np.array(f(np.array(x))).tolist()

alpha= 0.5

#%%

for i in range(N+1):
    
    if i == 0: # FOR INITAL DATASET
            print("initial Data")
            σ_1 = exponential_cov(x, x, θ) # get new Cov for x Data but same params
            
            x_pred = np.linspace(-3, 3, 1000) # linspace for the gaussian
            predictions = [predict(i, x, exponential_cov, θ, σ_1, y) for i in x_pred] 
            y_pred, sigmas = np.transpose(predictions)
            
            sigmas_normalized=np.divide(sigmas,sigmas.max())
            
            #plt.figure(21)
            #plt.plot(x_pred,y_pred)
            fig, axes = plt.subplots(4, 1, figsize=(6, 5))
            axes[0].errorbar(x_pred, y_pred, yerr=sigmas)
            axes[0].plot(x, y, "ro")
            #axes[0].plot(xpts,f(np.array(xpts)))
            axes[1].set_title("Gaussian Process")
            axes[1].plot(x_pred, sigmas_normalized)
            axes[1].set_title("Aquisition Function")
            #plt.savefig('init_5_datapoints.png')
            
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx] # RETURN x WErt
        
        
            for i in np.arange(len(x_pred)):
                if i == 0:
                    nn_first_value = find_nearest(x,x_pred[i])
                    nn_value = [f(nn_first_value)]
                else:
                   nn_value.append(f(find_nearest(x,x_pred[i])))
        
            dist = (nn_value - y_pred)**2
            dist_normalized = dist/dist.max()
            
            
            axes[2].set_title("DIST ")
            plt.plot(x_pred,dist_normalized) #+2*sigmas) Mit sigma angepasst
            axes[2].set_title("EIGF, y Term")
            axes[3].plot(x_pred,alpha*dist_normalized+(1-alpha)*sigmas_normalized) #+2*sigmas) Mit sigma angepasst
            axes[3].set_title("Combined with alpha = {}".format(alpha))
            
    else:
            print("Iteration Number: {}".format(i))
            # HIER MAX AUSWÄHLEN; ABER ANPASSEN !
            df = pd.DataFrame(data=(zip(x_pred,2*sigmas)),columns=['x_pred','sigma'])
            x_new = df.loc[df['sigma'].idxmax()].x_pred
            x.append(x_new)
            y_new = f(x_new)
            y.append(y_new)
            
            σ_new = exponential_cov(x, x, θ)
            predictions = [predict(i, x, exponential_cov, θ, σ_new, y) for i in x_pred]
            y_pred, sigmas = np.transpose(predictions)
            #sigma_max= df['sigma'].max()
            #sigmas = sigmas/ sigma_max
            sigmas_normalized=np.divide(sigmas,sigmas.max())

            fig, axes = plt.subplots(4, 1, figsize=(6, 5))
            axes[0].errorbar(x_pred, y_pred, yerr=sigmas) # y pred is MEAN
            axes[0].plot(x, y, "ro")
            #axes[0].plot(xpts,f(np.array(xpts)))
            axes[0].plot(x_pred, y_pred,'black',linestyle ='--',linewidth= 2)
            axes[1].set_title("Gaussian Process")
            axes[1].plot(x_pred, sigmas_normalized)
            axes[1].set_title("Aquisition Function")
            
            
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx] # RETURN x WErt
        
        
            for i in np.arange(len(x_pred)):
                if i == 0:
                    nn_first_value = find_nearest(x,x_pred[i])
                    nn_value = [f(nn_first_value)]
                else:
                   nn_value.append(f(find_nearest(x,x_pred[i])))
        
            dist = (nn_value - y_pred)**2
            dist_normalized = dist/dist.max()
            
            
            axes[2].set_title("EIGF, y Term")
            axes[2].plot(x_pred,dist_normalized) #+2*sigmas) Mit sigma angepasst

            #plt.savefig('iteration_{}.png'.format(i))
            

            axes[3].plot(x_pred,alpha*dist_normalized+(1-alpha)*sigmas_normalized) #+2*sigmas) Mit sigma angepasst
            axes[3].set_title("Combined with alpha = {}".format(alpha))

#%%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx] # RETURN x WErt


for i in np.arange(len(x_pred)):
    if i == 0:
        nn_first_value = find_nearest(x,x_pred[i])
        nn_value = [f(nn_first_value)]
    else:
        nn_value.append(f(find_nearest(x,x_pred[i])))

dist = (nn_value - y_pred)**2
dist_normalized = dist/dist.max()

fig, axes = plt.subplots(4, 1, figsize=(6, 5))
axes[0].errorbar(x_pred, y_pred, yerr=sigmas)
axes[0].plot(x, y, "ro")
#axes[0].plot(x_pred, y_pred,'black',linewidth=3)
#axes[0].plot(xpts,f(np.array(xpts)))

axes[1].set_title("Gaussian Process")
axes[1].plot(x_pred, sigmas_normalized)
axes[1].set_title("Aquisition Function")

axes[2].set_title("DIST ")
axes[2].plot(x_pred,dist_normalized) #+2*sigmas) Mit sigma angepasst
axes[2].set_title("EIGF, y Term")


axes[3].plot(x_pred,alpha*dist_normalized+(1-alpha)*sigmas_normalized) #+2*sigmas) Mit sigma angepasst
axes[3].set_title("Combined with alpha = {}".format(alpha))
    
    