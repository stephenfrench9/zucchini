import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim


class oneOperation(torch.nn.Module):
    def __init__(self, inputd, hiddend, outputd):
        super(oneOperation, self).__init__()
        self.mat1 = torch.nn.Linear(inputd, hiddend, bias=False)
        self.mat2 = torch.nn.Linear(hiddend, outputd, bias=False)

    def forward(self, x):
        """
        write this function as if x is a single trial
        """
        h_relu = self.mat1(x).clamp(min=0)
        y_pred = self.mat2(h_relu)
        return y_pred


def dummy_function(grass):
    first = grass[:5]
    second = grass[5:10]
    first_sum = torch.cumsum(first[0:5],dim=0)
    a = first_sum[-1]*first[4]
    second_sum = torch.cumsum(second[0:5],dim=0)
    b = second_sum[-1]*second[4]
    return torch.tensor([a.item(), b.item()])


def handle_all_trials(x, N, din, dowt):
    raw_tensor_object = torch.tensor((), dtype=torch.float32)
    holder = raw_tensor_object.new_ones((N, dowt))

    for inx, row in enumerate(x):
        holder[inx, :] = holder[inx,:]*dummy_function(row)

    return holder
        
        
if __name__ == "__main__":
    N, din, dowt, dhidden = 1000, 10, 2, 5
    epochs = 500

    g = oneOperation(din, dhidden, dowt)
    x = torch.randn(N, din)
    y = handle_all_trials(x, N, din, dowt)

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

        testx = torch.randn(N, din)
        testy = handle_all_trials(testx, N, din, dowt)
        testy_pred = g(testx)
        testoutput = loss(testy_pred, testy)

        if epoch%100 == 0:
            llog.append(round(output.item(), 2))
            testllog.append(round(testoutput.item(), 2))
            l = str(round(output.item(), 2))
            lt = str(round(testoutput.item(), 2))
            print('{:<15}:{}'.format('Training Error', l))
            print('{:<15}:{}'.format('Testing Error', lt))
            print()


    plt.figure(1)
    plt.title("train, " + str(N) + " trials")
    plt.xlabel("input dimension: " + str(din))
    plt.ylabel(str(epochs) + " epochs")
    plt.plot(llog)

    plt.figure(2)
    plt.title("test, " + str(N) +" trials")
    plt.xlabel("input dimension: " + str(din))
    plt.ylabel(str(epochs) + " epochs")
    plt.plot(testllog)
    plt.show()
## make a new customized module class. This class can be used to apply a linear transformation to a pytorch tensor.

## why subclass the module class? Why not just use the Linear class to get a linear transformation object and then apply the transformation? Same effect.

## well ... now you have a pytorch module with your customized behavior. There are many other pytorch objects/classes/functions that are designed to interact with a module. 
