import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N, din = 1000, 10
    randos = torch.randn(N, din)
    x = randos[:,0]
    y = randos[:,1]

    randos2 = torch.randn(N, din)
    x2 = randos2[:,0]
    y2 = randos2[:,1]

    plt.title("1000 random 10d vectors projected into 2 dimensions." +
              "\nTwo seperate batches.")
    plt.scatter(x, y)
    plt.scatter(x2, y2, marker = '.', s=10)

    
    plt.show()
    
