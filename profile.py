from fit import handle_all_trials
from fit import train
from fit import twoTransformations
from scipy.stats import linregress

import matplotlib.pyplot as plt
import torch

def init_weights(m):
    print("Here Is One Module:")
    print("Type: ")
    print(type(m))
    print()
    print(m.weight.data)
    print("now modify it")
    torch.Tensor.normal_(m.weight.data)
    print(m.weight.data)
    input("Ready for the next one?")
    if type(m) == torch.nn.Linear:
        print("We got a weird type match!!!")
    #    m.weight.data.fill_(1.0)
    #    print(m.weight)


def fit_data(N, din, dowt, dhidden, lr, epochs, device):

    print("Fittinig a Brand Shiny New Shiny shiny new neural net")
# apply(init_weights)
    f = twoTransformations(din, dhidden, dowt)
    x = torch.randn(N, din, device = device)
    y = handle_all_trials(x, N, din, dowt, device)

    llog, testllog = train(f, x, y, epochs, lr, N, din, dowt, device, True)
    slope = linregress(range(len(testllog)), testllog)
    
    return slope.slope


def run_all(sizes, din, dowt, dhidden, lr, epochs, device):
    """
    Find the test loss slope for each N in sizes
    """
    slopes = []
    for inx, N in enumerate(sizes):
       slope =  fit_data(N, din, dowt, dhidden, lr, epochs, device)
       slopes.append(slope)
       print("NN Trained********************Overall Progress: {}"
             .format(round((inx+1)/len(sizes),2)))
    return slopes

       
if __name__ == '__main__':
    """
    finds the slope for all input shapes
    """
    din, dhidden, dowt = 10, 5, 2
    d = twoTransformations(din, dhidden, dowt)
#    d.cuda()
    
#    device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    din, dowt = 10, 2 #Data shape
    dhidden = 5 #neural network architechture
    lr, epochs = .001, 5001 #Training parameters
   
    sizes = [25*i+25 for i in range(3)]

    slopes = run_all(sizes, din, dowt, dhidden, lr, epochs, device)

    plt.plot(sizes, slopes)
    plt.title("Test Loss Slope vs. Number of Trials\nlr: {}, epochs: {}"
              .format(lr, epochs))
    plt.xlabel("Trials: x, din: {}, dowt: {}".format(din, dowt))
    plt.ylabel("Slope: y, dhidden: {}".format(dhidden))
    plt.savefig("thisis")
    
