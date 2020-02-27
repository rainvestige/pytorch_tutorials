# coding=utf-8
''' ADVERSARIAL EXAMPLE GENERATION
    Research is constantly pushing ML models to be more faster, more efficient.
    However, an often overlooked aspect of designing and training models is 
    security and robustness, especially in the face of an adversary who wishes 
    to fool the model.

    This tutorial will raise yout awareness to the security vulnerabilites of
    ML models, and will give insight into the hot topic of adversarial 
    machine learning. You may be superised to find that adding imperceptible
    perturbations to an image can cause drastically different model performance.
    Given that this is a tutoria, we will explore the topic via example on an 
    image classifier. Specifically we will use one of the first and most 
    popular attack methods, the Fast Gradient Sign Attack(FGSM), to fool an
    MNIST classifier.
'''

''' Threat Model
    For context, there are many categories of adversarial attacks, each with a
    different goal and assumption of the attacker's knowledge. However, in general 
    the overarching goal is to add the least amount of perturbation to the input
    data to cause the desired misclassification. There are several kinds of 
    assumptions of the attacker's knowledge, two of which are: white-box and 
    black-box. A white-box attack assumes that the attacker has full knowledge
    and access to the model, including architecture, inputs, outputs, and weights.
    A black-box attack assumes the attacker only has access to the inputs and 
    outputs of the model, and knows nothing about the underlying architecture or 
    weights. There are also several types of goals, including misclassification 
    and source/target misclassification. A goal of misclassification means the
    adversary only wants the output classification to be wrong but does not care
    what the new classification is. A source/target misclassification means the
    adversary wants to alter an image that is originally of a specific  source class 
    so that it is classified as a specific target class.

    In this case, the FGSM attack is a white-box attack with the goal of 
    misclassification. With this background information, we can now discuss the 
    attack in detail.
'''

''' Faster Gradient Sign Attack
    One of the first and most popular adversarial attacks to date is referred to
    as the Fast Gradient Sign Attack(FGSM) and is described by Goodfellow et.al.
    in <<Explaining and Harnessing Adversarial Examples. The attck is remarkably
    powerful, and yet intuitive. It is designed to attack neural networks by 
    leveraging the way they learn, gradients. The idea is simple, rather than 
    working to minimize the loss by adjust the weights based on the backpropagated
    gradients, the attack adjusts the input data to **maximize the loss** based on the 
    same backpropagated gradients. In other words, the attack use the gradient of
    the loss with respect to(w.r.t) the input data, then adjusts the input data
    to maxmize the loss.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


''' Inputs
    There are only three inputs for this tutorial, and are defined as follows:
    * epsilons - List of epsilon values to use for the run. It is important to 
      keep 0 in the list because it represents the model performance on the 
      original test set. Also, intuitively we would expect the larger the epsilon,
      the more noticeable the perturbations but the more effective the attack in
      terms of degrading model accuracy. Since the data range here is [0, 1], 
      no epsilon value should exceed 1.
    * pretrianed_model - path to the pretrained MNIST model which was trained 
      with pytorch/examples/mnist. For simplicity, download the pretrained 
      model here
      (https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing).
    * use_cuda - boolean flag to use CUDA if desired and available. Note, a GPU with
    CUDA is not critical for this tutorial as a CPU will not take much time.
'''
epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = 'data/lenet_mnist_model.pth'
use_cuda = True


''' Model Under Attack
    As mentioned, the model under attack is the same MNIST model from 
    pytorch/examples/mnist. You may train and save your own MNIST model or you
    can download and use the provided model. The net defination and dataloader
    here have been copied from the MNIST example. The purpose of this section 
    is to define the model and dataloader, then initialize the model and load
    the pretrained weights.
'''
# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
transform = transforms.Compose([
    transforms.ToTensor(),])
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# define what device we are using
device = torch.device('cuda:0' if (use_cuda and torch.cuda.is_available()) else 'cpu')

# Initialize the network
model = Net().to(device)

# load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()


''' FGSM Attack
    Now, we can define the function that creates the adversarial examples by
    perturbing the original inputs. The `fgsm_attack` function takes three
    inputs, image is the original clean image(x), epsilon is the pixel-wise
    perturbation amount(e), and data_grad is gradient of the loss w.r.t the
    input image(Delta_x(J(theta,x,y))). The function then creates perturbed
    image as 
        perturbed_image=image+epsilon∗ sign(data_grad)=x+ϵ∗ sign(∇xJ(θ,x,y))
        
    Finally, in order to maintain the original range of the data, the perturbed
    image is clipped to range [0, 1]
'''
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0, 1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


''' Tessting Function
    Each call to this test function performs a full test step on the MNIST test
    set and reports a final accuracy. However, notice that this function also 
    takes an epsilon input. This is because the `test` function reports the
    accuracy of a model that is under attack from an adversary with strength e.
    More specifically, for each sample in the test set, the function computes
    the gradient of the loss w.r.t the input data(data_grad), creates a 
    perturbed image with fgsm_attack(perturbed_data), then checks to see if 
    the perturbed example is adversarial. In addition to testing the accuracy of
    the model, the function also saves and returns some successful adversarial
    examples to be visualized later.
'''
def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max
                                                   # log-probability
        # If the initial prediction is wrong, don't bother attacking, just
        # move on. Otherwise, we will implement the attack.
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward() 

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the pertubed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5 :
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print('Epsilon: {}\tTest Accuracy = {} / {} = {}'.format(
        epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples


''' Run Attack
    Here we run a full test step for each epsilon value in the epsilons input.
    Foreach epsilon we also save the final accuracy and some successful 
    adversarial examples to be plotted in the coming sections. Notice how the
    printed accyracies decrease as the epsilon value increases. Also, note
    the `e=0` case represents the original test accuracy, with no attack.
'''
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


''' Results(Accuracy vs Epsilon)
    The first result is the accyracy versus epsilon plot. As alluded to 
    earlier, as epsilon increases we expect the test accuracy to decrease.
    This is because larget epsilons mean we take a larget step in the
    direction that will maximize the loss. Notice the trend in the curve is 
    not linear even through the epsilon values are linearly spaced. For example,
    the accuracy at `e=0.05` is only about 4% lower than `e=0`, but the 
    accuracy at `e=0.2` is 25% lower than `e=0.15`. Also, notice the accuracy
    of the model hits random accyracy for a 10-class classifier between 
    `e=0.25` and `e=0.3`
'''
plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title('Accuracy vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.show()
    
        
''' Sample Adversarial Examples
    In this casem as epsilon increases the test accuracy decreases, but the 
    perturbation become more easily perceptible. In reality, there is a 
    tradeoff between accuracy degredation and perceptibility that an attacker
    must consider. Here, we show some examples of successful adversarial 
    examples at each epsilon value. Each row of the plot shows a different
    epsilon value. The title of each image shows the "original classification->
    adversarial classification". Notice, the perturbations start to become
    evident at `e=0.15` and are quite evident at `e=0.3`. However, in all
    cases humans are still capable of identifing the correct class despite the
    added noise.
'''
# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel('Eps: {}'.format(epsilons[i]), fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title('{} -> {}'.format(orig, adv))
        plt.imshow(ex, cmap='gray')
plt.tight_layout()
plt.show()
input('Press any key to stop')

