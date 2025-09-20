import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


def num_params(model):
            return np.sum([np.prod(x.shape) for x in model.parameters()])

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv_bn1 = self.create_conv_bn(3, 16, 7, 4, device, dtype)
        self.conv_bn2 = self.create_conv_bn(16, 32, 3, 2, device, dtype)

        f1 = nn.Sequential(*[
            self.create_conv_bn(32, 32, 3, 1, device, dtype),
            self.create_conv_bn(32, 32, 3, 1, device, dtype)
        ])
        self.res1 = nn.Residual(f1)

        self.conv_bn3 = self.create_conv_bn(32, 64, 3, 2, device, dtype)
        self.conv_bn4 = self.create_conv_bn(64, 128, 3, 2, device, dtype)

        f2 = nn.Sequential(*[
            self.create_conv_bn(128, 128, 3, 1, device, dtype),
            self.create_conv_bn(128, 128, 3, 1, device, dtype)
        ])
        self.res2 = nn.Residual(f2)

        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.res1(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = self.res2(x)
        x = nn.Flatten()(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION

    def create_conv_bn(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        model = []
        model.append(nn.Conv(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype))
        model.append(nn.BatchNorm2d(dim=out_channels, device=device, dtype=dtype))
        model.append(nn.ReLU())
        return nn.Sequential(*model)


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_layer = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model=='rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.prediction_layer = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embedded_x = self.embedding_layer(x)
        h_out, h = self.seq_model(embedded_x, h)
        seq_len, bs, hidden_size = h_out.shape
        h_out = h_out.reshape((seq_len*bs, hidden_size))
        out = self.prediction_layer(h_out)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)