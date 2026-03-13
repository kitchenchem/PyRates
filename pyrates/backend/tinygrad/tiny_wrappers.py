# Wraps tinygrad tensor methods for module style interface with PyRates codegen
from tinygrad import Tensor

def abs(x): return x.abs()
def exp(x): return x.exp()
def log(x): return x.log()
def sqrt(x): return x.sqrt()
def sign(x): return x.sign()
def round(x): return x.round()
def sigmoid(x): return x.sigmoid()

def sin(x): return x.sin()
def cos(x): return x.cos()
def tan(x): return x.tan()
def tanh(x): return x.tanh()
def sinh(x): return x.sinh()
def cosh(x): return x.cosh()
def arctan(x): return x.atan()
def arcsin(x): return x.asin()
def arccos(x): return x.acos()

def sum(x, **kwargs): return x.sum(**kwargs)
def mean(x, **kwargs): return x.mean(**kwargs)
def argmin(x, **kwargs): return x.argmin(**kwargs)

def maximum(x, y): return x.maximum(y)
def minimum(x, y): return x.minimum(y)
def matmul(x, y): return x.matmul(y)

def roll(x, shifts, dims=0): return x.roll(shifts, dims)
def concat(*args, dim=0): return Tensor.cat(*args, dim=dim)

def randn(*shape): return Tensor.randn(*shape)
