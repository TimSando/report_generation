import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import load


class ReportModel(nn.Module):
    def __init__(self, vocab_size, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.emb_layer = nn.Embedding(vocab_size, 200)

        ## define the LSTM
        self.lstm = nn.LSTM(200, n_hidden, n_layers, dropout=drop_prob, batch_first=True)

        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        """Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`."""

        ## pass input through embedding layer
        embedded = self.emb_layer(x)

        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)

        ## pass through a dropout layer
        out = self.dropout(lstm_output)

        # out = out.contiguous().view(-1, self.n_hidden)
        out = out.reshape(-1, self.n_hidden)

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if torch.cuda.is_available():
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
            )

        # if GPU is not available
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            )

        return hidden


class ModelTrainer:
    def __init__(
        self,
        input_array,
        target_array,
    ):
        self.input_array = input_array
        self.target_array = target_array

    def get_batches(self, input_array, target_array, batch_size: int = 32):
        prv = 0
        for n in range(batch_size, input_array.shape[0], batch_size):
            x = input_array[prv:n, :]
            y = target_array[prv:n, :]
            prv = n
            yield x, y

    def train(self, model, epochs=10, batch_size=32, lr=0.001, clip=1, print_every=32):

        # optimizer
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # loss
        criterion = nn.CrossEntropyLoss()

        # push model to GPU
        model.cuda()

        counter = 0

        model.train()

        for e in range(epochs):

            # initialize hidden state
            h = model.init_hidden(batch_size)

            for x, y in self.get_batches(self.input_array, self.target_array, batch_size):
                counter += 1

                # convert numpy arrays to PyTorch arrays
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                # push tensors to GPU
                inputs, targets = inputs.cuda(), targets.cuda()

                # detach hidden states
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                model.zero_grad()

                # get the output from the model
                output, h = model(inputs, h)

                # calculate the loss and perform backprop
                loss = criterion(output, targets.view(-1))

                # back-propagate error
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), clip)

                # update weigths
                opt.step()

                if counter % print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, epochs), "Step: {}...".format(counter))