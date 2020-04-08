# coding=utf-8
"""
    This is the third and final tutorial on doing "NLP From Scratch", where we
    write our own classes and functions to preprocess the data to do our NLP 
    modeling tasks. We hope after you complete this tutorial that you'll 
    proceed to learn how torchtext can handle much of this preprocessing for
    you in the three tutorials immediately following this one.

    In this project we will be teaching a neural network to translate from 
    French to English.

    This is made possible by the simple but powerful idea of the `sequence to
    sequence network`, in which two recurrent neural networks work together to
    transform one sequence to another. An encoder network condenses an input
    sequence into a vector, and a decoder network unfolds that vector into a
    new sequence.

    To improve upon this model we'll use an `attention mechanism, which lets
    the decoder learn to focus over a specific range of the input sequence.
"""


from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import random
import time
import math

from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading data files
# Download the data from https://download.pytorch.org/tutorial/data.zip and
# extract it to the current directory.
# 
# Similar to the character encoding used in the character-level RNN tutorial,
# we will be representing each word in a language as a one-hot vector, or 
# giant vector of zeros except for a single one(at the index of the word).
# Compared to the dozens of characters that might exist in a language, there
# are many many more words, so the encoding vector is much larger. We'll 
# however cheat a bit and trim the data to only use a few thousand words per 
# language.
# 
# We'll need a unique index per word to use as the inputs and targets of the
# networks later. To keep track of all this we will use a helper class called
# `Lang` which has word->index(word2index) and index->word(index2word) 
# dictionaries, as we as a count of each word (word2count) to use to later 
# replace rare words.


SOS_TOKEN = 0
EOS_TOKEN = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # start from 2(Count SOS and EOS)

    def add_sentence(self, sentence):
        """Creating word->index and index->word dictionaries from sentence"""

        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """Creating word->index and index->word dictionaries from word"""

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.n_words += 1
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
        else:
            self.word2count[word] += 1

    
# The files are all in Unicode, to simplify we will turn Unicode characters to 
# ASCII make everything lowercase, and trim most punctuation.

def unicode_to_ascii(s):
    """Turn a Unicode string to plain ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):

    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r'\1', s)  # Substitute
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def read_langs(lang1, lang2, reverse=False):
    """Read the data from the file in line format.

    To read the data file we will split the file into lines, and then split lines
    into pairs. The files are all English->Other languages, so if we want to
    translate from other language->English. We should added the `reverse` flag
    to reverse the pairs.

    Args:
        lang1: The language we will translate from.
        lang2: The language we will translate to.
        reverse: The flag used to reverse the pairs.

    """
    print('Reading lines...')

    # Read the file and split into lines
    lines = open('data/{}-{}.txt'.format(lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# Since there are a lot of example sentences and we want to train something 
# quickly, we'll trim the data set to only relatively short and simple
# sentences. Here the maximum length is 10 words(that includes ending 
# punctuation) and we're filtering to sentences that translate to the form 
# "I am" or " He is" etc.

MAX_LENGTH = 10

eng_prefix = (
    'i am', 'i m',
    'he is', 'he s',
    'she is', 'she s',
    'you are', 'you re',
    'we are', 'we re',
    'they are', 'they re'
)


def filter_pair(p):
    """The condition whether the pair should be filtered"""
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefix)

def filter_pairs(pairs):
    """Filter the pairs"""
    return [pair for pair in pairs if filter_pair(pair)]


# The full process for preparing the data is:
#    - Read text filt and split into lines, split lines into pairs
#    - Normalize text, filter by length and content
#    - Make word lists from sentences in pairs

def prepare_data(lang1, lang2, reverse=False):
    """Prepare the data"""

    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print('Read {} sentence pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs)
    print('Trimmed to {} sentence pairs'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
print(random.choice(pairs))


# The Seq2Seq Model
# A Recurrent neural network, or RNN, is a network that operates on a sequence
# and uses its own output as input for subsequent steps.
# A Sequence to Sequence network, or seq2seq network, or Encoder Decoder network,
# is a model consisting of two RNNs called the encoder and decoder. The encoder
# reads an input sequence and outputs a single vector, and the decoder reads that
# vector to produce an output sequence.
#
# Unlike sequence prediction with a single RNN, where every input corresponds 
# to an output, the seq2seq model frees us from sequence length and order, which
# makes it ideal for translation between two languages.
#
# Consider the sentence "Je ne suis pas le chat noir" -> "I am not the black cat".
# Most of the words in the input sentence have a direct translation in the 
# output sentence, but are in slightly different orders, e.g. "chat noir" and
# "black cat". Because of the "ne/pas" construction there is also one more word
# in the input sentence. It would be difficult to produce a correct translation
# directly from the sequence of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the ideal
# case, encodes the "meaning" of the input sequence into a single vector--a
# single point in some N dimensional space of sentences.

# The Encoder
# The encoder of a seq2seq network is a RNN that outputs some value for every 
# word from the input sentence. For every input word the encoder outputs a 
# vector and a hidden state, and uses the hidden state for the next input word.

class EncoderRnn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)


    def forward(self, input_, hidden):
        embedded = self.embedding(input_).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

# The Decoder
# The decoder is another RNN that takes the encoder output vector(s) and outputs
# a sequence of words to create the translation.
#
# Simple Decoder
# In the simplest seq2seq decoder we use only last output of the encoder. This
# context vector is used as the initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and hidden
# state. The initial input token is the start-of-string<SOS> token, and the 
# first hidden state is the context vector(the encoder's last hidden state).


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        output = self.embedding(input_).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
        

# Attention Decoder
# If only the context vector is passed between the encoder and decoder, that
# single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to 'focus' on a different part of the 
# encoder's outputs for every step of the decoder's own outputs. First we
# calculate a set of attention weights. These will be multiplied by the encoder
# output vectors to create a weighted combination. The result(called
# `attn_applied` in the code) should contain information about the specific 
# part of the input sentence, and thus help the decoder choose the right output
# words.
#
# Calculating the attention weights is done with another feed-forward layer 
# `attn`, using the decoder's input and hidden state as inputs. Because there
# are sentences of all sizes in the training data, to actually create the train
# this layer we have to choose a maximum sentence length(input length, for
# encoder outputs) that it can apply to. Sentences of the maximum length will
# use all the attention weights,  while shorter sentences will only use the 
# first few.


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input_, hidden, encoder_outputs):
        embedded = self.embedding(input_).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), 
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


# Training
# Preparing Training Data
# To train, for each pair we will need an input tensor(indexes of the words in
# the input sentence) and target tensor(indexes of the words in the target 
# sentence). While creating these vectors we will append the EOS token to both
# sequences.


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# Training the Model
# To train we run the input sentence through the encoder, and keep track of 
# every output and the lateset hidden state. Then the decoder is given the 
# <SOS> token as its first input, ant the last hidden state of the encoder as
# its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as each
# next input, instead of using the decoder's guess as the next input. Using
# teacher forcing causes it to converge faster but `when the trained network is
# exploited, it may exhibit instablity`.
#
# You can observe outputs of teacher-forced networks that read with coherent
# grammar but wander far from the correct translation-intuitively it has 
# learned to represent the output grammar and can "pick up" the meaning once
# the teacher tells it the first few words, but it has not properly learned how
# to create the sentence from the translation in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly choose to
# use teacher forcing or not with a simple if statement. Turn 
# `teacher_forcing_ratio` up to use more of it.

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[EOS_TOKEN]], device=DEVICE)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


    if use_teacher_forcing:
        # Teacher forcing: Feed teh target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = topi.squeeze().detach()   # Detach from history as input

            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# This is a helper function to print time elapsed and estimated time remaining
# given the current time and progress %.

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s （- %s)' % (as_minutes(s), as_minutes(rs))


# The whole training process looks like this:
# - Start a timer
# - Initialize optimizers and criterion
# - Create set of training pairs
# - Start empty losses array for plotting
#
# Then we call `train` many times and occasionally print the progress(% of 
# examples, time so fat, estimated time) and average loss.

def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
                learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset every print_every
    plot_loss_total = 0  # reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss 

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, i/n_iters),
                                         i, i/n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


# Plotting results
# Plotting is done with matplotlib, using the array of loss values `plot_losses`
# saved while training.

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# Evaluation
# Evaluation is mostly the same as training, but there are no targets so we 
# simply feed the decoder's predictions back to itself for each step. Every 
# time it predicts a word we add it to the output string, and if it predicts
# the EOS token we stop there. We also store the decoder's attention outputs 
# for display later.

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hiddden_size,
                                      device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[EOS_TOKEN]], devce=DEVICE)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attention = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attention[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attention[:di + 1]

# We can evaluate random sentences from the training set and print out the 
# input, target, and output to make some subjective quality judgements:

def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# Training and Evaluating
# With all these helper functions in place(it looks like extra work, but it 
# makes it easier to run multiple experiments) we can actually initialize
# a network and start training

# Remember that the input sentences were heavily filtered. For this small 
# dataset we can use relatively small networks of 256 hidden nodes and a single
# GRU layer. After about 40 minutes on a MacBook CPU we'll get some reasonable
# results.

hidden_size = 256
encoder1 = EncoderRnn(input_lang.n_words, hidden_size).to(DEVICE)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)\
                .to(DEVICE)

train_iters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluate_randomly(encoder1, attn_decoder1)

# Visualizing Attention
# A useful property of the attention mechanism is its highly interpretable 
# outputs. Because it is used to weight specific encoder outputs of the input
# sequence, we can imagine looking where the network is focused most at each 
# time step.

# You could simply run `plt.matshow(attentions)` to see attention output 
# displayed as a matrix, with the columns being input steps and rows being
# output steps:

output_words, attentions = evaluate(encoder1, attn_decoder1, 'je suis trop froid .')
plt.matshow(attentions.numpy())


# For a better viewing experience we will do the extra work of adding axes and 
# labels:

def show_attention(input_sentence, output_words, attentions):
    # set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


evaluate_and_show_attention('elle a cinq ans de moins que moi.')
evaluate_and_show_attention('elle est trop petit .')
evaluate_and_show_attention('je ne crains pas de mourir .')
evaluate_and_show_attention('c est un jeune directeur plein de talent .')

