# coding=utf-8
'''
    We are still hand-crafting a small RNN with a few linear layers. The big 
    difference is instead of predicting a category after reading in all the
    letters of a name, we input a category and output one letter at a time.
    Recurrently predicting characters to form language(this could also be 
    done with words or other higher order constructs) is often referred to as 
    a "language model".
'''


''' Preparing the Data
    Download the data from here (https://download.pytorch.org/tutorial/data.zip)
    and extract it to the correct directory

    See the last tutorial for more detail of this processs. In short, there 
    are a bunch of plain text files `data/names/[Languages].txt` with a name
    per line. We split lines into an arrary, convert Unicode to ASCII, and end
    up with a dictionary `{language: [names...]}
'''

import random
import glob
import os
import unicodedata
import string
from io import open

import torch
import torch.nn as nn

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker
print('length of letters: {}\n'.format(n_letters))

def find_files(path):
    return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category(language)
category_lines = {}
all_categories = []
for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]  # obtain the * in *.txt
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found.')

print('# categories: ', n_categories, all_categories)
print(unicode2ascii("O'Néàl"))
print('########################################\n')


''' Creating the Network
    This network extends the last tutorial's RNN with an extra argument for the 
    category tensor, which is concatenated along with the others. The category
    tensor is a one-hot vector just like the letter input.

    We will interpret the output as the probability of the next letter. When 
    sampling, the most likely output letter is used as the next input letter.

    I added a second linear lay `o2o`(after combining hidden and output) to 
    give it more muscle to work with. There's also a dropout layer, which 
    randomly zeros parts of its input with a given probability(here 0.1)
    and is usually used to fuzz  inputs to prevent overfitting. Here we're
    using it towards the end of the network to purposely add some chaos
    and increase sampling variety.
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category_, input_tensor_, hidden_tensor):
        input_combined = torch.cat((category_, input_tensor_, hidden_tensor), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


''' Preparing for Training
    First of all, helper functions to get random pairs of (category, lines)
'''

# Random item from a list
def random_choice(list_):
    return list_[random.randint(0, len(list_) -1)]

# Get a random category and random line from that category
def random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line

'''
    For each timestep(that is, for each letter in a training word) the inputs of
    the network will be (category, current_letter, hidden_state) and the outputs
    will be (next letter, next hidden state). So for each training set, we'll 
    need the category, a set of input letters, and a set of output/target 
    letters.

    Since we are predicting the next letter from the current letter for each 
    timestep, the letter pairs are groups of consecutive letters from the line 
    -e.g. for "ABCD<EOS>" we would create ("A", "B"), ("B", "C"), ("C", "D")
    , ("D", "EOS").

    THe category tensor is a one-hot tensor of size <1 x n_categories>. When
    training we feed it to the network at every timestep - this is a design
    choice, it could have been included as part of initial hidden state or
    some other strategy.
'''

# One-hot vector for category
def category_tensor(category):
    idx = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][idx] = 1
    return tensor

# One-hot matrix of first to last letters(not including EOS) for input
def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        idx = all_letters.find(line[li])
        tensor[li][0][idx] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
'''
    For convenience during training we'll make a `random_training_example` 
    function that fetches a random (category, line) pair and turns them into the
    required(category, input, target) tensors.
'''

# Make category, input, and target tensors from a random category, line pair
def random_training_example():
    category, line = random_training_pair()
    category_tensor_ = category_tensor(category)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return category_tensor_, input_line_tensor, target_line_tensor


''' Training the network
    In constast to  classification, where only the last output is used, we are 
    making a prediction at every step, so we are calculating loss at every step.

    The magic of autograd allows you to simply sum these losses at each step 
    and call backward at the end.
'''

criterion = nn.NLLLoss()
lr = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()
    
    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

# To keep track of how long training takes, i am adding a `timeSince(timestamp)`
# function which returns a human readable string:

import time
import math

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Training is business as usual - call train a bunch of times and wait a few 
# minutes, printing current time and loss every `print_every` examples, and 
# keeping store of an average loss per `plot_every` examples in `all_losses`
# for plotting later.


rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0  # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*random_training_example())
    total_loss += loss

    if iter % print_every == 0:
        print('{} ({} {:.2f}%) {:.4f}'.format(time_since(start), iter, 
            iter/n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


''' Plotting the Results
    Plotting the historical loss from `all_losses` shows the network learning
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
    

''' Sampling the Network
    To sample we give the network a letter and ask what the next one is, feed
    that in as the next letter, and repeat until the EOS token.

    * Create tensors for input category, starting letter, and empty hidden state
    * Create a string `output_name` with the starting letter
    * Up to a maximum output length,
        - Feed the current letter to the network
        - Get the next letter from highest output, and next hidden state
        - If the letter is EOS, stop here
        - If a regular letter, add to `output_name` and continue
    * Return the final name
'''

max_length = 20
# sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor_ = category_tensor(category)
        input_tensor_ = input_tensor(start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor_, input_tensor_[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input_tensor_ = input_tensor(letter)
       
    return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')
samples('German', 'GER')
samples('Spanish', 'SPA')
samples('Chinese', 'CHI')
