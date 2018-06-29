from fit import train
from fit import twoTransformations
from fit import handle_all_trials
from fit import dummy_function
import matplotlib.pyplot as plt
import torch


#def fff(ff, fff) {
#} does that look like python? Python uses indent to indicate code blocks

            

if __name__ == "__main__":
    N, din, dowt, dhidden = 1000, 10, 2, 5
    epochs = 5000

    r = twoTransformations(din, dhidden, dowt)
    f = twoTransformations(din, dhidden, dowt)
    x = torch.randn(N, din)
    y = handle_all_trials(x, N, din, dowt)

#    train(f, x, y, epochs)

    testn = 1000

    zoom = torch.randn(testn, din)
    zoomt = handle_all_trials(zoom, testn, din, dowt)
    zoomtr = r(zoom)
    zoomtf = f(zoom)

    truex = zoomt.detach().numpy()[:,0]
    truey = zoomt.detach().numpy()[:,1]

    rx = zoomtr.detach().numpy()[:,0]
    ry = zoomtr.detach().numpy()[:,1]

    fx = zoomtf.detach().numpy()[:,0]
    fy = zoomtf.detach().numpy()[:,1]



    plt.scatter(truex, truey, marker='8')
    plt.scatter(rx, ry, marker='x')
    plt.scatter(fx, fy, marker='+')
    plt.show()


    
 
#   go ahead and assume that you have your two models. 
    
#    display_weights(r, f)

#    x = torch.randn(N, din)
#    y = handle_all_cases(x)
    
#    yr = r(x)
#    yf = f(x)

   # plot(y, yr)
   # plot(y, yf)

    
