from fit import train
from fit import twoTransformations
from fit import handle_all_trials
import matplotlib.pyplot as plt
import torch


#def fff(ff, fff) {
#} does that look like python? Python uses indent to indicate code blocks

            

if __name__ == "__main__":
    N, din, dowt, dhidden = 1000, 10, 2, 5
    epochs = 500

    r = twoTransformations(din, dhidden, dowt)

    f = twoTransformations(din, dhidden, dowt)
    x = torch.randn(N, din)
    y = handle_all_trials(x, N, din, dowt)
    
    train(f, x, y, epochs)
#   go ahead and assume that you have your two models. 
    
#    display_weights(r, f)

 #   x = torch.randn(N, din)
  #  y = handle_all_cases(x)
    
   # yr = r(x)
    #yf = f(x)

   # plot(y, yr)
   # plot(y, yf)

    
