"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*self.scalar*power_scalar(node.inputs[0], self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        lhs, rhs = divide(out_grad, b), -1*multiply(out_grad, divide(a, power_scalar(b, 2)))
        return lhs, rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = (-2, -1)
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if len(self.axes) == 2:
            new_axes = [i for i in range(len(a.shape))]
            new_axes[self.axes[0]] = new_axes[self.axes[0]]^new_axes[self.axes[1]]
            new_axes[self.axes[1]] = new_axes[self.axes[0]]^new_axes[self.axes[1]]
            new_axes[self.axes[0]] = new_axes[self.axes[0]]^new_axes[self.axes[1]]
            new_axes = tuple(new_axes)
        else:
            new_axes = self.axes

        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if len(self.axes) == 2:
            return transpose(out_grad, self.axes)

        new_axes = [-1 for _ in self.axes]
        for axis, old_pos in enumerate(self.axes):
            new_axes[old_pos] = axis
        return transpose(out_grad, tuple(new_axes))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        reverse_shape = node.inputs[0].shape
        return reshape(out_grad, reverse_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        sum_axis = []
        if len(node.inputs[0].shape) == 0:
            return summation(out_grad)
        elif len(node.inputs[0].shape) == 1 and node.inputs[0].shape[0] == 1:
            return reshape(summation(out_grad), (1, ))
        else:
            assert len(out_grad.shape) == len(node.inputs[0].shape)
            for i in range(len(out_grad.shape)):
                if node.inputs[0].shape[i] != out_grad.shape[i]:
                    assert node.inputs[0].shape[i] == 1
                    sum_axis.append(i)
        sum_axis = tuple(sum_axis)
        out = reshape(summation(out_grad, axes=sum_axis), node.inputs[0].shape)
        return out
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None or type(self.axes)==int or len(self.axes)==1:
            if type(self.axes) == int and self.axes<0:
                self.axes += len(a.shape)
            elif type(self.axes) == tuple and self.axes[0] < 0:
                self.axes = self.axes[0]+len(a.shape)
            return array_api.summation(a, axis=self.axes)
        
        new_ordered_axes = []
        new_shape = []
        last_size = 1
        for axis, shape in enumerate(a.shape):
            if not axis in self.axes:
                new_ordered_axes.append(axis)
                new_shape.append(shape)
            else:
                last_size *= shape
        new_ordered_axes = new_ordered_axes + list(self.axes)
        new_shape.append(last_size)
        a = a.permute(new_ordered_axes).compact()
        a = a.reshape(new_shape).compact()
        return array_api.summation(a, axis=len(new_shape)-1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            final_shape = []
            initial_shape = out_grad.shape
            idx = 0
            axes = tuple([self.axes]) if type(self.axes) is int else self.axes
            for i in range(len(node.inputs[0].shape)):
                if (i in axes) or (i-len(node.inputs[0].shape) in axes):
                    final_shape.append(1)
                else:
                    final_shape.append(initial_shape[idx])
                    idx += 1
            out_grad_reshape = out_grad.reshape(tuple(final_shape))
        else:
            out_grad_reshape = out_grad.reshape(tuple([1 for _ in range(len(node.inputs[0].shape))]))
        out = broadcast_to(out_grad_reshape, node.inputs[0].shape)
        return out
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        lhs = matmul(out_grad,transpose(b, (-1, -2)))
        rhs = matmul(transpose(a, (-1, -2)),out_grad)

        if (len(a.shape) != len(lhs.shape)):
            axes = tuple([i for i in range(len(lhs.shape)-len(a.shape))])
            lhs = summation(lhs, axes=axes)

        if (len(b.shape) != len(rhs.shape)):
            axes = tuple([i for i in range(len(rhs.shape)-len(b.shape))])
            rhs = summation(rhs, axes=axes)
        return lhs, rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1*out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return divide(out_grad, a)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * relu(a)/a
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        _Z_max = Z.max(axis=self.axes)
        # if self.axes is not None:
        new_shape = list(_Z_max.shape)
        if self.axes is not None:
            axes = [self.axes] if type(self.axes) == int else list(self.axes)
            for axis in axes:
                new_shape.insert(axis, 1)
            new_shape = tuple(new_shape)
            _Z_max = _Z_max.compact()
        else:
            new_shape = tuple([1 for _ in range(len(Z.shape))])
        Z_max = _Z_max.reshape(new_shape)
        Z_max = array_api.broadcast_to(Z_max, Z.shape).compact()

        return array_api.log(array_api.summation(array_api.exp(Z-Z_max), axis=self.axes)) + _Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z_max = Tensor(Z.data.realize_cached_data().max(axis=self.axes), device=Z.device)
        new_shape = list(Z_max.shape)
        if self.axes is not None:
            axes = [self.axes] if type(self.axes) == int else list(self.axes)
            for axis in axes:
                new_shape.insert(axis, 1)
            new_shape = tuple(new_shape)
            Z_max = Z_max
        else:
            new_shape = tuple([1 for _ in range(len(Z.shape))])
        Z_max = Z_max.reshape(new_shape)
        Z_max = broadcast_to(Z_max, Z.shape)

        _sum_exp = summation(exp(Z-Z_max), axes=self.axes)
        _log_sum_exp_derivative_to_sum_exp = out_grad / _sum_exp

        if self.axes is not None:
            final_shape = []
            initial_shape = out_grad.shape
            idx = 0
            axes = tuple([self.axes]) if type(self.axes) is int else self.axes
            for i in range(len(Z.shape)):
                if i in axes:
                    final_shape.append(1)
                else:
                    final_shape.append(initial_shape[idx])
                    idx += 1
            out_grad_reshape = _log_sum_exp_derivative_to_sum_exp.reshape(tuple(final_shape))
        else:
            out_grad_reshape = _log_sum_exp_derivative_to_sum_exp
        _log_sum_exp_derivative_to_exp = broadcast_to(out_grad_reshape, Z.shape)
        

        exp_derivative_to_Z = exp(Z-Z_max)

        return _log_sum_exp_derivative_to_exp*(exp_derivative_to_Z)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        return out_grad * (1-(tanh(z)**2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        # calculate the shape for output
        out_shape = list(args[0].shape)
        out_shape.insert(self.axis, len(args))
        out_shape = tuple(out_shape)

        out_tensor = NDArray.make(out_shape, device=args[0].device)
        start_slice = 0
        for arg_idx in range(len(args)):
            arg = args[arg_idx]
            curr_slice = [slice(0, end, 1) for end in out_shape]
            curr_slice[self.axis] = slice(start_slice, start_slice+1, 1)
            start_slice += 1
            out_tensor.__setitem__(idxs=tuple(curr_slice), other=arg)
        return out_tensor
        ## END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        final_grads = split(out_grad, axis=self.axis)
        return final_grads
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out_shape = list(A.shape)
        num_out_tensor = out_shape[self.axis]
        out_shape = [out_shape[i] for i in range(len(out_shape)) if i != self.axis]
        out_shape = tuple(out_shape)

        # initialize all output
        out = []
        for start in range(num_out_tensor):
            curr_slice = [slice(0, size, 1) for size in out_shape]
            curr_slice.insert(self.axis, slice(start, start+1, 1))
            out_tensor = A.__getitem__(tuple(curr_slice)).compact().reshape(out_shape)
            out.append(out_tensor)
        out = tuple(out)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        final_grad = stack(out_grad, axis=self.axis)
        return final_grad
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = []
        for axis, _ in enumerate(new_shape):
            step = 1
            if axis in self.axes:
                new_shape[axis] *= (self.dilation+1)
                step = self.dilation+1
            slices.append(slice(0, new_shape[axis], step))
        out = array_api.full(shape=new_shape, fill_value=0, dtype=a.dtype, device=a.device)
        out.__setitem__(tuple(slices), a)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = []
        for axis, _ in enumerate(a.shape):
            step = self.dilation+1 if axis in self.axes else 1
            slices.append(slice(0, a.shape[axis], step))
        return a.__getitem__(tuple(slices))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        axes_pad = tuple( 
            [
                (self.padding, self.padding) if axis==1 or axis==2 else (0, 0) 
                for axis in range(len(A.shape)) 
            ] 
        )
        A = A.pad(axes_pad)
        n_samples, height, width, in_channel = A.shape
        kernel_size, _, in_channel, out_channel = B.shape

        out_shape = (n_samples, (height-kernel_size)//self.stride+1, (width-kernel_size)//self.stride+1, out_channel)
        out = array_api.full(shape=out_shape, fill_value=0, dtype=A.dtype, device=A.device)

        for k1 in range(kernel_size):
            for k2 in range(kernel_size):
                a = A[:, k1:(k1+height-kernel_size+1):self.stride, k2:(k2+width-kernel_size+1):self.stride, :].compact()
                a = a.reshape((-1, in_channel))
                b = B[k1, k2, :, :].compact().reshape((in_channel, out_channel))
                out += (a@b).reshape(out.shape)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        n_samples, height, width, in_channel = A.shape
        kernel_size, _, in_channel, out_channel = B.shape
        axes_pad = tuple( 
            [
                (self.padding, self.padding) if axis==1 or axis==2 else (0, 0) 
                for axis in range(len(A.shape)) 
            ] 
        )
        B_flip = transpose(flip(B, axes=(0,1)),(0,1,3,2))
        dilated_out_grad = dilate(out_grad, (1, 2), dilation=self.stride-1)

        A_grad = conv(dilated_out_grad, B_flip, 1, kernel_size-1)
        A_grad = Tensor(A_grad.detach().cached_data[:, self.padding:self.padding+height, self.padding:self.padding+width, :].compact(), device=A.device, dtype=A.dtype)

        _A = transpose(A, (3, 1, 2, 0))
        dilated_out_grad = transpose(dilated_out_grad, (1, 2, 0, 3))

        B_grad = transpose(conv(_A, dilated_out_grad, 1, self.padding), (1, 2, 0, 3))

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



