# coding=utf-8
'''
    Neural networks can be constructed using the `torch.nn` package.

    A typical training precedure for a neural network is as follows:
    1. Define the neural network that has some learnable parameters(or weight)
    2. Iterate over a dataset of inputs
    3. Process input through the network
    4. Comput the loss(how far is the output from being correct)
    5. Propagate gradients back into the network's parameters
    6. Updata the weights of the network, typically using a simple update rule:
       `weight = weight - learning_rate * gradient`
'''

'''Define the network '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) # change shape(flat)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s # why?
        return num_features


net = Net()
print(net)

print('------------------------------------------------------------')
# The learnable parameters of a model are returned by `net.parameters()`
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

print('------------------------------------------------------------')
# Let's try a random 32x32 input. Note: expected input size of this net is 
# 32x32. To use this net on the MNIST dataste, please resize the images 
# from the dataset to 32x32.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


print('------------------------------------------------------------')
# Zero the gradient buffers of all parameters and backprops with random 
# gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

''' Note
    torch.nn only supports mini-batches. The entire `torch.nn` package only
    supports inputs that are a mini-batch of samples, and not a single sample.
    
    For example, `nn.Conv2d` will take in a 4D Tensor of 
    |nSamples x nChannels x Height X Width|

    If you have a single sample, just use `input.unsqueeze(0)` to add a fake
    batch dimension
'''

'''
   - torch.Tensor - A multi-dimensional array with support for autograd operations like 
     backward(). Also holds the gradient w.r.t. the tensor.
   - nn.Module - Neural network module. Convenient way of encapsulating parameters, with helper
     s for moving them to GPU, exporting, loading, etc.
    - nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when 
      assigned as an attribute to a Module.
    - autograd.Function - Implements forward and backward definitions of an autograd operation.
      Every Tensor operation creates at least a single Function node that connects to functions
      that created a Tensor and encodes its history.
'''

''' Loss Function
    A loss function takes the (output, target) pair of inputs, and computes a 
    value that estimates how far away the output is from the target.

    There are several different loss functions under the nn package. A simpe 
    loss is: `nn.MSELoss` which computes the mean-squared error between the 
    input and the target.
'''
output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# So, when we call `loss.backward()`, the whole graph is differentiated w.r.t.
# the loss, and all Tensors in the graph that has `requires_grad=True` will
# have their `.grad` Tensor accumulated with the gradient.
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU


print('------------------------------------------------------------')
''' Backprop
    To backpropagate the error al we have to do is to `loss.backward()`. You 
    need to clear the existing gradients though, else gradients will be 
    accumulated into the existing gradients.
'''
# Now we shall call `loss.backward()`, and have a look at conv1's bias gradients
# before and after the backward.
net.zero_grad()     # zeros the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


print('------------------------------------------------------------')
''' Update the weights
    The simplest update rule used in practice is the Stochastic Gradient Descent(SGD)
    `weight = weight - learning_rate * gradient
'''
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# However, as you use neural networks, you want to use various different update
# rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this, we 
# built a small package:`torch.optim` that implements all these methods. Using
# it is very simple
import torch.optim as optim

# Create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
