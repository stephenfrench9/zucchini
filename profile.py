from fit import handle_all_trials
from fit import train
from fit import twoTransformations
from scipy.stats import linregress

import matplotlib.pyplot as plt
import torch

def fit_data(N, din, dowt, dhidden, lr, epochs):

    f = twoTransformations(din, dhidden, dowt)
    x = torch.randn(N, din)
    y = handle_all_trials(x, N, din, dowt)

    llog, testllog = train(f, x, y, epochs, lr, N, din, dowt, True)
    slope = linregress(range(len(testllog)), testllog)
    
    return slope.slope


def run_all(sizes, din, dowt, dhidden, lr, epochs):
    """
    Find the test loss slope for each N in sizes
    """
    slopes = []
    for inx, N in enumerate(sizes):
       slope =  fit_data(N, din, dowt, dhidden, lr, epochs)
       slopes.append(slope)
       print("NN Trained********************Overall Progress: {}"
             .format(round((inx+1)/len(sizes),2)))
    return slopes

       
if __name__ == '__main__':
    din, dowt = 10, 2 #Data shape
    dhidden = 5 #neural network architechture
    lr, epochs = .001, 5001 #Training parameters
    
    sizes = [25*i+25 for i in range(3)]

    slopes = run_all(sizes, din, dowt, dhidden, lr, epochs)

    plt.plot(sizes, slopes)
    plt.title("Test Loss Slope vs. Number of Trials\nlr: {}, epochs: {}"
              .format(lr, epochs))
    plt.xlabel("Trials: x, din: {}, dowt: {}".format(din, dowt))
    plt.ylabel("Slope: y, dhidden: {}".format(dhidden))
    plt.savefig("thisis")
    
