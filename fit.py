import matplotlib.pyplot as plt
from scipy.stats import linregress
import torch
from torch import nn
from torch import optim


class twoTransformations(torch.nn.Module):
    def __init__(self, inputd, hiddend, outputd):
        super(twoTransformations, self).__init__()
        self.mat1 = torch.nn.Linear(inputd, hiddend, bias=False)
        self.mat2 = torch.nn.Linear(hiddend, outputd, bias=False)

    def forward(self, x):
        """
        write this function as if x is a single trial
        """
        h_relu = self.mat1(x).clamp(min=0)
        y_pred = self.mat2(h_relu)
        return y_pred


def dummy_function(grass, device):

    first = grass[:5]
    second = grass[5:10]
    first_sum = torch.cumsum(first[0:5],dim=0)
    a = first_sum[-1]*first[4]
    second_sum = torch.cumsum(second[0:5],dim=0)
    b = second_sum[-1]*second[4]
    return torch.tensor([a.item(), b.item()], device = device)


def handle_all_trials(x, N, din, dowt, device):

    raw_tensor_object = torch.tensor((), dtype=torch.float32, device=device)
    holder = raw_tensor_object.new_ones((N, dowt))

    for inx, row in enumerate(x):
        holder[inx, :] = holder[inx,:]*dummy_function(row, device)

    return holder


def train(g, x, y, epochs, lr, N, din, dowt, device, disp_prog):
    loss = nn.MSELoss()
    optimizer = optim.SGD(g.parameters(), lr=.001)
    llog = []
    testllog = []
    
    for epoch in range(epochs):
        y_pred = g(x)
        output = loss(y_pred, y)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()

        N = x.size()[0]
        din = x.size()[1]
        dowt = y.size()[1]
        
        testx = torch.randn(N, din, device = device)
        testy = handle_all_trials(testx, N, din, dowt, device)
        testy_pred = g(testx)
        testoutput = loss(testy_pred, testy)

        if epoch%100 == 0:
            llog.append(round(output.item(), 2))
            testllog.append(round(testoutput.item(), 2))
            l = str(round(output.item(), 2))
            lt = str(round(testoutput.item(), 2))
            if disp_prog:
                print('{:<15}:{}'.format('Training Error', l))
                print('{:<15}:{}'.format('Testing Error', lt))
                print('progress: {}'.format(round(epoch/epochs,2)))
                print()

    return llog, testllog


def display(llog, testllog, epochs, lr, N, din):
    domain = range(len(testllog))
    slope,  inter,  r,  p,  err  = linregress(domain, llog)
    slopet, intert, rt, pt, errt = linregress(domain, testllog)

    slope, inter = round(slope, 3), round(inter, 3)
    slopet, intert = round(slopet, 3), round(intert, 3)
    
    plt.figure(1)
    plt.title("Training Loss" )
    plt.xlabel("slope: {}, intercept: {}".format(slope, inter))
    plt.ylabel("{} epochs, {} learn rate\n, {} trials, {} inputs"
               .format(epochs, lr, N, din))
    plt.plot(domain, slope*domain + inter)
    plt.plot(llog)


    plt.figure(2)
    plt.title("Testing Loss" )
    plt.xlabel("slope: {}, intercept: {}".format(slopet, intert))
    plt.ylabel("{} epochs, {} learn rate\n {} trials, {} inputs"
               .format(epochs, lr, N, din))
    plt.plot(domain, slopet*domain + intert)
    plt.plot(testllog)

    plt.show()
    
        
if __name__ == "__main__":
    #    device = torch.device("cuda:0")
    device = torch.device("cpu")

    N, din, dowt, dhidden = 1000, 10, 2, 5
    epochs, lr = 201, .001

    g = twoTransformations(din, dhidden, dowt)
    x = torch.randn(N, din)
    y = handle_all_trials(x, N, din, dowt)

    llog, testllog = train(g, x, y, epochs, lr, N, din, dowt, device, True)

    display(llog, testllog, epochs, lr, N, din)

## make a new customized module class. This class can be used to apply a linear transformation to a pytorch tensor.

## why subclass the module class? Why not just use the Linear class to get a linear transformation object and then apply the transformation? Same effect.

## well ... now you have a pytorch module with your customized behavior. There are many other pytorch objects/classes/functions that are designed to interact with a module. 
