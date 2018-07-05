from fit import handle_all_trials
from fit import train
from fit import twoTransformations
from scipy.stats import linregress

import matplotlib.pyplot as plt
import numpy as np
import torch

def slope(N, din, dowt, dhidden, lr, epochs, device):
    """
    generate new model, generate new data. Fit it.
    """
    f = twoTransformations(din, dhidden, dowt)
    f.cuda()

    x = torch.randn(N, din, device = device)
    y = handle_all_trials(x, N, din, dowt, device)

    llog, testllog = train(f, x, y, epochs, lr, N, din, dowt, device, True)
    slope = linregress(range(len(testllog)), testllog)
    
    return slope.slope


if __name__ == '__main__':
    """
    Finds the distribution of slopes. 
    """
    din, dowt = 10, 2 #Data shape
    dhidden = 5 #neural network architechture
    lr, epochs = .001, 5001 #Training parameters
    device = torch.device("cuda:0") #choose hardware

    N = 200
    slopes = []
    for N in range(10):
        N = N + 1
        print(N)
        rv = slope(N, din, dowt, dhidden, lr, epochs, device)
        slopes.append(rv)
        print(slopes)
    print("The average is: ")    
    print(sum(slopes,0)/1000)
    
    plt.hist(slopes)
    plt.title('Distribution of slopes for N = 200')
    plt.xlabel('slope')
    plt.ylabel('frequency')
    plt.savefig('slopeDist')
