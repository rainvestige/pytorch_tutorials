# coding=utf-8
''' DCGAN TUTORIAL
    Introduction
    This tutorial will give an introduction to DCGANS through an example. We
    will train a generative adversarial network(GAN) to generate new celebrities
    after showing it pictures of many real celebrities. Most of the code here is 
    from the dcgan implementation in `pytorch/examples(https://github.com/
    pytorch/examples)`, and this document will give a thorough explaination of
    the implementation and shed light on how and why this model works. But don't 
    worry, no prior knowledge of GANs is required, but it may require a 
    first-timer to spend some time reasoning about what is actually happening
    under the hood. Also, for the sake of time it will help to have a GPU, or
    two.
'''

''' What is a GAN?
    GANs are a framework for teaching a DL model to capture the training data's
    distribution so we can generate new data from the same distribution. GANs
    were invented by lan Goodfellow in 2014 and first described in the paper
    `Generative Adversarial Nets`
    (https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) 
    They are mode of two distinct models, a generator and a discriminator. The
    job of the generator is to spawn 'fake' images that look like the training
    images. The job of the discriminator is to look at an image and output 
    whether or not it is a real training image or a fake image from the 
    generator. During training, the generator is constantly trying to outsmart
    the discriminator by generating better and better fakes, while the 
    discriminator is working to become a better detective and correctly 
    classify the real and fake images. The equilibrium of this game is when the
    generator is generating perfect fakes that look as if they came directly
    from the training data, and the discriminator is left to always guess 50%
    confidence that the generator output is real or fake.

    Now lets define some notation to be used throughout tutorial starting with
    the discriminator. Let `x` be data repersenting an image. `D(x)` is the 
    discriminator network which outputs the scalar probability that `x` came
    from the training data rather than generator. Here, since we are dealing
    with images, the input to `D(x)` is an image of CxHxW size 3x64x64.
    Intutively, `D(x)` should be HIGH when `x` comes from training data and
    LOW when `x` comes from the generator. `D(x)` can also be thought of as
    a traditional binary classifier.

    For the generator's natation, let `z` be a latent space vector sampled from 
    a standard normal distribution. `G(z)` represents the generator function
    which maps the latent vector `z` to data space. The goal of `G` is to
    estimate the distribution that the training data comes from (Pdata) so it
    can generate fake samples from that estimated distribution(Pg).

    SO, `D(G(z))` is the probability scalar that the output of the generator G
    is a real image. As described in "Goodfellow's paper", D and G play a
    minimax game in which `D` tries to maximize the probability it correctly
    classifies reals and fakes(log(D(x))), and `G` tries to minimize the 
    probability that `D` will predict it's outputs are fake(log(1-D(G(x)))).
    From the paper, the GAN loss function is
            min_G && max_D{V(G,D)} = E[logD(x)] + E[log(1-D(G(z)))]
    In theory, the solution to this minimax game is where Pg = Pdata, and the
    discriminator guesses randomly if the inputs are real or fake. However,
    the convergence theory of GANs is still being actively researched and in
    reality models do not always train to this point.
'''

''' What is a DCGAN?
    A DCGAN is a direct extension of the GAN described above, except that it
    explicitly uses convolutional and convolutional-transpose layers in
    in the discriminator and generator, respectively. It was first described
    by Radford et.al. in the paper `Unsupervised Repersentation Learning 
    With Deep Convolutional Generative Adversarial Networks`(https://arxiv.org/
    pdf/1511.06434.pdf). The discriminator is made up of strided convolution 
    layers, batch norm layers, and LeakyReLU activations. The input is a 
    3x64x64 input image and the output is a scalar probability that the input
    is from the real data distribution. The generator is comprised of 
    convolutional-transpose layers, batch norm layers, and ReLU activations.
    The input is a latent vector, z, that is drawn from a standard normal 
    distribution and the output is a 3x64x64 RGB image. The strided conv-
    transpose layers allow the latent vector to be transformed into a volume
    with the same shape as an image. In this paper, the authors also give
    some tips about how to setup the optimizers, how to calculate the loss
    functions, and how to initialize the model weights, al of which will be
    explained in the coming sections.
'''

#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms 
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


''' INPUT ''' 
# root dir for dataset
dataroot = '/usr/local/download/data/'

# number of workers for dataloader
num_workers = 2

# batch size during training
batch_size = 64

# spatial size of training images. All images will be resized to this size
# using a transformer.
image_size = 64

# number of channels in the training images. For color images this is 3
num_channels = 3

# size of `z` latent vector (i.e. size of generator input)
size_z = 100

# size of feature maps in generator 
size_generator_feat = 64

# size of feature maps in discriminator 
size_discriminator_feat = 64

# number of training epochs
num_epochs = 5

# learning rate for optimizers
lr = 0.0002

# beta1 hyperparam for adam optimizers
beta1 = 0.5

# number of GPUS available. Use 0 for CPU model
num_gpus = 1


''' Data
    In this tutorial we will use the `Celeb-A Faces dataset` which can be
    downloaded at the linked site, or in `Google Drive`. The dataset wil
    download as a file named img_align_celeba.zip. Once download, create
    a directory named celeba and extract the zip file into that directory.
    Then, set the dataroot input for this notebook to the celeba directory
    you just created. The resulting directory structure should be:
    /path/to/celeba
        -> img_align_celeba
            -> 188242.jpg
            -> 173822.jpg
               ...
    This is an important step because we will be using the ImageFolder dataset
    class, which requires there to be subdirectories in the dataset's root 
    folder. Now, we can create the dataset, dataloader, set the device to run
    on, and finally visualize some of the training data.
'''
# we can use an image folder dataset the way we have it setup.
# Create the dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.ImageFolder(root=dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

device = torch.device('cuda:0' if (torch.cuda.is_available() and num_gpus > 0)
        else 'cpu')

# Plot some training images
print('Plotting some training images')
real_batch = next(iter(dataloader)) # B x C x H x W
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title('Training images')

plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), 
            (1, 2, 0))
)
plt.show()


''' Implementation
    With our input parameters set and the dataset prepared, we can now get into 
    the implementation. We'll start with the weight initialization strategy,
    then talk about the generator, discriminator, loss functions, and training
    loop in detail.
'''

''' Weight Initialization
    From the DCGAN paper, the authors specify that all model weights shall be
    randomly initialized from a Normal distribution with mean=0, stdev=0.02.
    The `weights_init` function takes an initialized model as input and
    reinitializes all convolutional, convolutional-transpose, and batch 
    normalization layers to meet this criteria. This function is applied to 
    the models immediately after initialization.
'''
# custom weights initialization called on NetG and netD
def weight_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

''' Generator
    The generator, G, is designed to map the latent space vector(z) to 
    data-space. Since our data are images, converting `z` to data-space means
    ultimately create a RGB image with the same size as the training images(
    i.e. 3x64x64). In practice, this is accomplished through a series of 
    strided two dimensional convolutional transpose layers, each paired with
    a 2d batch norm layer and a relu activation. The output of the generator 
    is fed through a tanh function to return it to the input data range of 
    [-1,1]. It is worth noting the existence of the batch norm functions after
    the conv-transpose layers, as this is a critical contribution of DCGAN 
    paper. These layers help with the flow of gradients during training.  

    Notice, the how the inputs we set in the input section(size_z,
    size_generator_feat, and num_channels) influence the generator architecture
    in code. size_z is the length of the `z` input vector, size_generator_feat
    relates to the size of the feature maps that are propagated through the
    generator, and num_channels is the number of num_channels in the output image
    (set to 3 for RGB images). Below is the code for the generator.
'''

# Generator Code
class Generator(nn.Module):
    def __init__(self, num_gpus):
        super(Generator, self).__init__()
        self.num_gpus = num_gpus
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(size_z, size_generator_feat*8, 4 ,1, 0,
                               bias=False),
            nn.BatchNorm2d(size_generator_feat*8),
            nn.ReLU(True),

            # state size. (sgf*8) x 4 x 4
            nn.ConvTranspose2d(size_generator_feat*8, size_generator_feat*4, 4 ,2, 1,
                               bias=False),
            nn.BatchNorm2d(size_generator_feat*4),
            nn.ReLU(True),

            # state size. (sgf*4) x 8 x 8
            nn.ConvTranspose2d(size_generator_feat*4, size_generator_feat*2, 4 ,2, 1,
                               bias=False),
            nn.BatchNorm2d(size_generator_feat*2),
            nn.ReLU(True),

            # state size. (sgf*2) x 16 x 16
            nn.ConvTranspose2d(size_generator_feat*2, size_generator_feat, 4 ,2, 1,
                               bias=False),
            nn.BatchNorm2d(size_generator_feat),
            nn.ReLU(True),

            # state size. (sgf) x 32 x 32
            nn.ConvTranspose2d(size_generator_feat, num_channels, 4, 2, 1, 
                               bias=False),
            nn.Tanh()
            #state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Now, we can instantiate the generator and apply the `weight_init` function. 
# check out the printed model to see how the generator object is structed.

# Create the generator
netG = Generator(num_gpus).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (num_gpus > 1):
    netG = nn.DataParallel(netG, list(range(num_gpus)))

# Apply the weights_init function to randomly initialize all weights to mean=0,
# stdev=0.2.
netG.apply(weight_init)

# Print the model
print(netG)


''' Discriminator
    As mentioned, the discriminator,`D`, is a binary classification network 
    that takes an image as input and outputs a scalar probability that the 
    input image is real(as opposed to fake). Here, `D` takes a 3x64x64 input 
    image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU
    layers, and outputs the final probability through a Sigmoid activation
    function. This architecture can be extended with more layers if necessary
    for the problem, but there is significance to the use of the strided 
    convolution, BatchNorm, and LeakyReLUs. The DCGAN paper mentions it is a 
    good practice to use strided convolution rather than pooling to downsample
    because it lets the network learn its own pooling function. Also batch norm
    and leaky relu functions promote healthy gradient flow which is critical 
    for the learning process of both `G` and `D`.
'''
class Discriminator(nn.Module):
    def __init__(self, num_gpus):
        super(Discriminator, self).__init__()
        self.num_gpus = num_gpus 
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(num_channels, size_discriminator_feat, 4, 2, 1, 
                bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (size_d_f) x 32 x 32
            nn.Conv2d(size_discriminator_feat, size_discriminator_feat * 2, 4, 2, 1, 
                bias=False),
            nn.BatchNorm2d(size_discriminator_feat * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (size_d_f*2) x 16 x 16
            nn.Conv2d(size_discriminator_feat * 2, size_discriminator_feat * 4, 4, 2, 1, 
                bias=False),
            nn.BatchNorm2d(size_discriminator_feat * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (size_d_f*4) x 8 x 8 
            nn.Conv2d(size_discriminator_feat * 4, size_discriminator_feat * 8, 4, 2, 1, 
                bias=False),
            nn.BatchNorm2d(size_discriminator_feat * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (size_d_f*8) x 4 x 4
            nn.Conv2d(size_discriminator_feat * 8, 1, 4 , 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Now, as with the generator, we can create the discriminator, applu the 
# `weight_init` function, and print the model's structure

# create the discirminator
netD = Discriminator(num_gpus).to(device)

# handle multi-gpu if desired
if (device.type == 'cuda') and (num_gpus > 1):
    netD = nn.DataParallel(netD, list(range(num_gpus)))

# apply the weight_init function to randomly initialize all weights to mean=0,
# stdev=0.2.
netD.apply(weight_init)

# print the model
print(netD)


''' Loss Functions and Optimizers
    WIth D and G setup, we can specify how they learn through the loss functions
    and optimizers. We will use the Binary Cross Entropy loss(BCELoss) function

    Notice how this function provides the calculation of both log components in
    the objective function(i.e. log(D(z)) and log(1-D(G(z)))). We can specify 
    what part of the BCE equation to use with `y` input. This is accomplished
    in the training loop which is coming up soon, but it is important to 
    understand how we can choose which component we wish to calculate just by
    changing `y`(i.e. GT labels).

    Next, we define our real label as 1 and the fake label as 0. These labels 
    will be used when calculating the loss of D and G, and this is also the
    convention used in the original GAN paper. Finally, we set up two separate
    optimizers, one for D and one for G. As specified in the DCGAN paper, both
    are Adam optimizers with learning rate 0.0002 and Beta1=0.5. For keeping
    track of the generator's learning pregression, we will generate a fixed 
    batch of latent vectors that are drawn from a Gaussian distribution(i.e.
    fixed_noise). In the training loop, we will periodically input this 
    fixed_noise into G, and over the iterations we will see images form out of
    the noise.
'''
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, size_z, 1, 1, device = device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Set up Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


''' Training
    Finally, now that we have all of the parts of the GAN framework defined, we
    can train it. Be mindful that training GANs is somewhat of an art form,as
    incorrect hyperparameter settings lead to mode collapse with little 
    explaination of what went wrong. Here, we will closely follow Algorithm 1
    from Goodfellow's paper, while abiding by some of the best practices shown 
    in ganhacks(https://github.com/soumith/ganhacks). Namely, we'll "construct
    different mini-batches for real and fake" images, and also adjust G's 
    objective function to maximize log(D(G(z))). Training is split up into two
    main parts. Part 1 undates the Discriminator and Part 2 updates the Generator.

    **Part1 - Train the Discriminator**
    Recall, the foal of training the discriminator is to maximize the probability
    of correctly classifying a given input as real or fake. In terms of Goodfellow,
    we wish to "update the discriminator by ascending its stochastic gradient".
    Practically, we want to maximize `log(D(x)) + log(1-D(G(z)))`. Due to the 
    separate mini-batch suggsetion from ganhacks, we will calculate this in 
    two steps. First, we will construct a batch of real samples from the training
    set, forward pass through `D`, calculate the loss(log(D(x))), then calculate
    the gradients in a backward pass. Secondly, we will construct a batch of 
    fake samples with the current Generator, forward pass this batch through D,
    calculate the loss (log(1-D(G(z)))) and accumulate the gradients with a
    backward pass. Now, with the gradients accumulated from both the all-real 
    and all-fake batches, we call a step of the discriminator's optimizer.

    **Part2 - Train the Generator**
    As stated in the original paper, we want to train the Generator by 
    minimize `log(1-D(G(z)))` in an effort to generate better fakes. As 
    mentioned, this was shown by Goodfellow to not provide sufficient gradients,
    especially early in the learning process. As a fix, we instead wish to 
    maximize `log(D(G(z)))`. In the code we accomplish this by: classifying the
    Generator output from Part1 with the Discriminator, computing G's loss
    using real labels as GT, computing G's gradients in a backward pass, and 
    finally updating G's parameters with an optimizer step. It may seem counter-
    intuitive to use the real labels as GT labels for the loss function, but
    this allows us to use the log(x) part of the BCELoss (rather than the log(1-x)
    part) which is exactly what we want.

    Finally, we will do some statistic reporting and at the end of each epoch 
    we will push our fixed_noise batch through the generator to visually track
    the progress of G's training. The training statistic reported are:
    * Loss_D - discriminator loss calculated as the sum of losses for the all
    real and all fake batches(log(D(x)) + log(D(G(z)))).
    * Loss_G - generator loss calculated as log(D(G(z)))
    * D(x) - the average output(across the batch) of the discriminator for the 
    all real batch. This should start close to 1 then theoretically converge 
    to 0.5 when G gets better. Think about why this is.
    * D(G(z)) - average discriminator outputs for the all fake batch. The first
    number is before D is updated and the second number is after D is updated.
    These numbers should start near 0 and converge to 0.5 as G gets better.
    Think about why this is.
'''

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print('Starting Training Loop...')
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ####################
        # (1) Update D network: maximize log(D(x)) + log(1-D(G(z)))
        ####################

        ## Train with all real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0) # batch_size
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, size_z, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake 
        # Update D
        optimizerD.step()


        ####################
        # (2) Update G network: maximize log(D(G(z)))
        ####################
        netG.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake 
        # batch through D
        output = netD(fake).view(-1)
        # calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Updata G
        optimizerG.step()

        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(), 
                      errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) \
        or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1


''' Results
    Finally, lets check out how we did. Here, we will look at three different 
    results. First, we will see how D and G’s losses changed during training. 
    Second, we will visualize G’s output on the fixed_noise batch for every 
    epoch. And third, we will look at a batch of real data next to a batch of 
    fake data from G.
'''

## Loss versus training iteration
plt.figure(figsize=(10, 5))
plt.title('Generator and Discriminator Loss During Training')
plt.plot(G_losses, label='G')
plt.plot(D_losses, label='D')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

## Visualization of G's progression
# Remember how we saved the generator's output on the fixed_noise batch after
# every epoch of training. Now, we can visualize the training progression of G 
# with an animation. Press the play button to start the animation.

#%%capture
fig = plt.figure(figsize=(8, 8))
plt.axis('off')
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000,
        blit=True)
HTML(ani.to_jshtml())


## Real Images vs. Fake Images
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Real Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], 
    padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Fake Images')
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()




