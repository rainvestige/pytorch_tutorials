# coding=utf-8
'''
    We will be building and training a basic character-level RNN to classify 
    words. This tutorial, along with the following two, show how to do 
    preprocess data for NLP modeling "from scratch", in particular not using
    many of the convenience functions of `torchtext`, so you can see how 
    preprocessing for NLP modeling works at a low level.

    A character-level RNN reads words as a series of character - outputting a
    prediction and "hidden state" at each step, feeding its previous hidden 
    state into each next step. We take the final prediction to be output, i.e. 
    which class the word belongs to.

    * Specially, we'll train on a few thousand surnames from 18 languages of 
    origin, and predict which language a name is from based on the spelling:
'''

''' Preparing the Data
    Download the data from (https://download.pytorch.org/tutorial/data.zip) and
    extract it to the current directory.

    included in the data/names directory are 18 text files named as 
    "[Language].txt". Each file contains a bunch of names, one name per line,
    mostly romanized(but we still need to convert from Unicode to ASCII).

    We'll end up with a dictionary of lists of names per language, 
    {language: [names ...]}. The generic variables "category" and "line"(for 
    language and name in our case) are used for later extensibility.
'''
from io import open
import glob
import os

def find_files(path): return glob.glob(path)

print(find_files('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
num_letters = len(all_letters)

# Turn a Unicode string to plain ASCII
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicode2ascii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]

for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines['English'][:10])


''' Turning Names into Tensors
    #####################################
    input format (Tensor)   
    #####################################
              0 1 2 3 ... num_letters
        0
        1
        .
        .
        .
    num_lines
    #####################################
    Now that we have all the names organized, we need to turn them into Tensors
    to make any use of them.

    To represent a single letter,  we use a 'one-hot vector' of size 
    <1 x num_letters>. A one-hot vector is filled with 0s except for a 1 at 
    index of the current letter, e.g. 'b' = <0 1 0 0 ...>, 'd' = <0 0 0 1 ...>

    To make a word we join a bunch of those into a 2D matrix 
    <line_length x 1 x num_letters>.

    That extra 1 dimension is because Pytorch assumes everything is in batches
    - we're just using a batch size of 1 here.
'''
import torch

# Find letter index form all_letters e.g. 'a' = 0 'b' = 1
def letter2index(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x num_letters> Tensor
def letter2tensor(letter):
    tensor = torch.zeros(1, num_letters)
    tensor[0][letter2index(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x num_letter>,
# or an array of one-hot letter vectors
def line2tensor(line):
    tensor = torch.zeros(len(line), 1, num_letters)  # `1` for batch size = 1
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor


print(letter2tensor('X'))

print(line2tensor('Xiao').size())


''' Creating the Network
    Before autograd, creating a recurrent neural network in Torch involved 
    cloning the parameters of a layer over several timesteps. The layers held
    hidden state and gradients whichare now entirely handled by the graph 
    itself. This mean you can implement a RNN in a very 'pure' way, as regular
    feed-forward layers.

    This RNN module is just 2 linear layers which operate on an input and 
    hidden state, with a LogSoftmax layer after the output.
'''
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        next_hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, next_hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

num_hidden = 128
rnn = RNN(num_letters, num_hidden, n_categories)

# To run a step of this network we need to pass an input(in our case, the Tensor
# for the current letter) and a previous hidden state(which we initialize as 
# zeros at first). We'll get back the output(probability of each language) and
# a next hidden state(which we keep for the next step)

# net_input = letter2tensor('X')
# hidden = torch.zeros(1, num_hidden)
# output, next_hidden = rnn(net_input, hidden)

net_input = line2tensor('Xiaoxinyu')
hidden  = torch.zeros(1, num_hidden) 
output, next_hidden = rnn(net_input[0], hidden)
print(output,'\n',next_hidden)


''' Training 

    Preparing for Training

    Before going into training we should make a few helper functions. The first
    is to interpret the output of the network, which we know to be a likelihood
    of each category. We can use `Tensor.topk` to get the index of the greatest
    value:
'''
def category_from_output(output):
    top_k, top_idx = output.topk(1)
    #top_k, top_idx = torch.topk(output[0], 1)
    category_idx = top_idx[0].item()
    return all_categories[category_idx], category_idx

print(category_from_output(output))

# We will also want a quick way to get a training example(a name and its 
# language):
import random

def random_choice(line):
    return line[random.randint(0, len(line)-1)]

def random_training_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], 
            dtype=torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_example()
    print('category = ', category, '/ line = ', line)

''' Training the Network
    Now all it takes to train this network is show it a bunch of example, have 
    it make guesses, and tell it if it's wrong. For the loss function 
    `nn.NLLLoss` is appropriate, since the last layer of the RNN is `nn.LogSoftmax`.
'''
criterion = nn.NLLLoss()

# Each loop of training will:
# 1. Create input and target tensors
# 2. Create a zeroed initial hidden state
# 3. Read each letter in and keep hidden state for next letter
# 4. Compare final output to target
# 5. Back-propagate
# 6. Return the output and loss

lr = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.item()

# Now we just have to run that with a bunch of examples. Since the `train` function
# returns both the output and loss we can print its guesses and also keep track of 
# loss for plotting. Since there are 1000s of examples we print only every
# `print_every` examples, and take an average of the loss.
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters+1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = 'true' if guess == category else 'false (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter/n_iters * 100,
            time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    

''' Plotting the Results
    Plotting the historical loss from `all_losses` shows the network learning
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

''' Evaluating the Results
    To see how well the network performs on different categories, we will 
    create a confusion matrix, indicating for every actual language(rows)
    which language the network guesses(columns). To calculate the confusion
    matrix a bunch of samples are run through the network with `evaluate()`,
    which is the same as `train()` minus the backprop.
'''
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_example()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i  = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


''' Running on User Input '''
def predict(input_line, n_predictions=3):
    print('\n> {}'.format(input_line))
    with torch.no_grad():
        output = evaluate(line2tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('({:.2f}) {}'.format(value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Xiaoxinyu')
predict('Satoshi')


