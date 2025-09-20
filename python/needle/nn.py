"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                self.in_features, self.out_features, device=device, dtype=dtype
            ),
            device=device, dtype=dtype)
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    self.out_features, None, device=device, dtype=dtype
                ),
                device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X@self.weight
        if self.use_bias:
            bias_reshape = ops.broadcast_to(ops.reshape(self.bias, (1, self.out_features)), out.shape)
            return out+bias_reshape
        else:
            return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1+ops.exp(-1*x))**(-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        _true_one_hot = init.one_hot(logits.shape[1], y, dtype=logits.dtype, device=logits.device).reshape(logits.shape)
        _z_y = ops.summation(_true_one_hot * logits, axes=-1)
        _log_sum_exp_z = ops.log(ops.summation(ops.exp(logits), axes=-1))
        losses = _log_sum_exp_z - _z_y
        return ops.summation(losses, axes=-1)/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_mean = 0
        self.running_var = 1
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            N = x.shape[0]
            mean = ops.summation(x, axes=0)/N
            broadcast_mean = ops.broadcast_to(ops.reshape(mean, (1, -1)), x.shape)
            var = ops.summation((x - broadcast_mean)*(x - broadcast_mean), axes=0)/N
            broadcast_var = ops.broadcast_to(ops.reshape(var, (1, -1)), x.shape)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
            broadcast_mean = ops.broadcast_to(ops.reshape(mean, (1, -1)), x.shape)
            broadcast_var = ops.broadcast_to(ops.reshape(var, (1, -1)), x.shape)

        norm_x = (x - broadcast_mean) / ops.power_scalar(broadcast_var+self.eps, 0.5)
        weight_reshape = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), norm_x.shape)
        bias_reshape = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), norm_x.shape)
        return weight_reshape * norm_x + bias_reshape
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        H = x.shape[1]
        mean = ops.summation(x, axes=1)/H
        mean = ops.broadcast_to(ops.reshape(mean, (-1, 1)), x.shape)
        var = ops.summation((x - mean)*(x - mean), axes=1)/H
        var = ops.broadcast_to(ops.reshape(var, (-1, 1)), x.shape)
        nominator = x - mean
        denominator = ops.power_scalar(var + self.eps, 0.5)
        return self.weight * nominator / denominator + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            seeds = init.randb(*x.shape, p=1-self.p)/(1-self.p)
            return seeds * x
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=kernel_size*kernel_size*in_channels,
            fan_out=kernel_size*kernel_size*out_channels,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = Parameter(
                init.rand(
                    *(out_channels, ),
                    low=-1.0/((self.in_channels * (kernel_size**2))**0.5),
                    high=1.0/((self.in_channels * (kernel_size**2))**0.5),
                    device=device,
                    dtype=dtype
                )
            )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(x, axes=(0, 2, 3, 1))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size//2)
        if self.bias is not None:
            out = out + ops.broadcast_to(ops.reshape(self.bias, (1, 1, 1, self.out_channels)), out.shape)
        out = ops.transpose(out, axes=(0, 3, 1, 2))
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1/hidden_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.W_ih = Parameter(
            init.rand(
                *(input_size, hidden_size),
                low=-k**0.5,
                high=k**0.5,
                device=device,
                dtype=dtype
            )
        )
        self.W_hh = Parameter(
            init.rand(
                *(hidden_size, hidden_size),
                low=-k**0.5,
                high=k**0.5,
                device=device,
                dtype=dtype
            )
        )
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    *(hidden_size, ),
                    low=-k**0.5,
                    high=k**0.5,
                    device=device,
                    dtype=dtype
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    *(hidden_size, ),
                    low=-k**0.5,
                    high=k**0.5,
                    device=device,
                    dtype=dtype
                )
            )
        self.activation = Tanh() if nonlinearity == 'tanh' else ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        b_ih_reshape, b_hh_reshape = 0, 0
        h = init.zeros(
            *(X.shape[0], self.hidden_size),
            device=self.device,
            dtype=self.dtype
        ) if h is None else h
        if self.bias_ih is not None and self.bias_hh is not None:
            b_ih_reshape = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
            b_hh_reshape = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))

        return self.activation(X@self.W_ih+b_ih_reshape+h@self.W_hh+b_hh_reshape)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = []
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        for i in range(num_layers):
            if i==0:
                self.rnn_cells.append(
                    RNNCell(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        bias=bias,
                        nonlinearity=nonlinearity,
                        device=device,
                        dtype=dtype
                    )
                )
            else:
                self.rnn_cells.append(
                    RNNCell(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        bias=bias,
                        nonlinearity=nonlinearity,
                        device=device,
                        dtype=dtype
                    )
                )
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h0 = init.zeros(
            *(self.num_layers, X.shape[1], self.hidden_size),
            device=self.device,
            dtype=self.dtype
        ) if h0 is None else h0

        h_t = [] # len = seq_len+1
        seq_len, bs, input_size = X.shape
        h_t.append(h0)
        for t in range(seq_len):
            inputs = ops.split(X, axis=0)[t]
            h_t_n = [] # len = num_layers
            for k in range(self.num_layers):
                h_out = self.rnn_cells[k](inputs, ops.split(h_t[t], axis=0)[k])
                h_t_n.append(h_out)
                inputs = h_out
            h_t.append(ops.stack(h_t_n, axis=0))
        h_t = ops.stack(h_t[1:], axis=0) # seq_len x num_layers x bs x hidden_size
        out = ops.split(h_t, axis=1)
        h_o = ops.split(h_t, axis=0)
        return out[self.num_layers-1], h_o[seq_len-1]
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1/hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.W_ih = Parameter(
            init.rand(
                *(input_size, 4 * hidden_size),
                low=-k ** 0.5,
                high=k ** 0.5,
                device=device,
                dtype=dtype
            )
        )
        self.W_hh = Parameter(
            init.rand(
                *(hidden_size, 4 * hidden_size),
                low=-k ** 0.5,
                high=k ** 0.5,
                device=device,
                dtype=dtype
            )
        )
        self.bias_ih, self.bias_hh = None, None
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    *(4 * hidden_size, ),
                    low=-k ** 0.5,
                    high=k ** 0.5,
                    device=device,
                    dtype=dtype
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    *(4 * hidden_size,),
                    low=-k ** 0.5,
                    high=k ** 0.5,
                    device=device,
                    dtype=dtype
                )
            )
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        b_ih_reshape, b_hh_reshape = 0, 0
        h = (
            init.zeros(
                *(X.shape[0], self.hidden_size),
                device=self.device,
                dtype=self.dtype
                ),
            init.zeros(
                *(X.shape[0], self.hidden_size),
                device=self.device,
                dtype=self.dtype
            )
        ) if h is None else h

        if self.bias_ih is not None and self.bias_hh is not None:
            b_ih_reshape = \
                self.bias_ih.reshape(
                    (1, 4 * self.hidden_size)
                ).broadcast_to((X.shape[0], 4 * self.hidden_size))
            b_hh_reshape = self.bias_hh.reshape(
                (1, 4 * self.hidden_size)
            ).broadcast_to((X.shape[0], 4 * self.hidden_size))

        h0, c0 = h
        # bs x 4*hidden_size
        Z = X@self.W_ih + b_ih_reshape + h0@self.W_hh + b_hh_reshape

        Z = Z.reshape((Z.shape[0], 4, self.hidden_size))
        i = self.sigmoid(ops.split(Z, axis=1)[0])
        f = self.sigmoid(ops.split(Z, axis=1)[1])
        g = self.tanh(ops.split(Z, axis=1)[2])
        o = self.sigmoid(ops.split(Z, axis=1)[3])

        c_out = f * c0 + i * g
        h_out = o * self.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = []
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells.append(
                    LSTMCell(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        bias=bias,
                        device=device,
                        dtype=dtype
                    )
                )
            else:
                self.lstm_cells.append(
                    LSTMCell(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        bias=bias,
                        device=device,
                        dtype=dtype
                    )
                )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h = (
            init.zeros(
                *(self.num_layers, X.shape[1], self.hidden_size),
                device=self.device,
                dtype=self.dtype
            ),
            init.zeros(
                *(self.num_layers, X.shape[1], self.hidden_size),
                device=self.device,
                dtype=self.dtype
            )
        ) if h is None else h
        h0, c0 = h
        seq_len = X.shape[0]
        h_t = [h0]
        c_t = [c0]
        for t in range(seq_len):
            inputs = ops.split(X, axis=0)[t]
            h_t_l = []
            c_t_l = []
            for k in range(self.num_layers):
                h_out, c_out = \
                    self.lstm_cells[k](
                        inputs,
                        (
                            ops.split(h_t[t], axis=0)[k],
                            ops.split(c_t[t], axis=0)[k]
                        )
                    )
                h_t_l.append(h_out)
                c_t_l.append(c_out)
                inputs = h_out
            h_t.append(ops.stack(h_t_l, axis=0))
            c_t.append(ops.stack(c_t_l, axis=0))
        h_t = ops.stack(h_t[1:], axis=0)
        c_t = ops.stack(c_t[1:], axis=0)
        out = ops.split(h_t, axis=1)[self.num_layers-1]
        h_o = ops.split(h_t, axis=0)[seq_len-1]
        c_o = ops.split(c_t, axis=0)[seq_len-1]
        return out, (h_o, c_o)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(
                *(num_embeddings, embedding_dim),
                mean=0,
                std=1,
                device=device,
                dtype=dtype
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_one_hot = init.one_hot(
            n=self.num_embeddings,
            i=x,
            device=self.device,
            dtype=self.dtype
        )
        seq_len, bs, _ = x_one_hot.shape
        x_one_hot = x_one_hot.reshape((seq_len*bs, self.num_embeddings))
        z = x_one_hot@self.weight
        return z.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION
