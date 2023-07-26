from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, cache
import random 
from typing import Callable

import numpy as np
from sympy import symbols, exp, log, sin, cos, lambdify

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from .base import MetadataBase

TRAIN = 'train'
VAL = 'val'

class Task(MetadataBase, ABC):
    def __init__(self, seed=0.):
        """
        seed: The seed to use for randomly generating examples of this task.
        """
        self.seed = seed

    @abstractmethod
    def get_dataset(self, i, type=TRAIN) -> Dataset:
        """
        Gets the dataset for the ith example of this task.
        """
        pass

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @property
    @abstractmethod
    def mi_output_shape(self):
        pass

    def get_metadata(self):
        return super().get_metadata() | {'seed': self.seed}


class Example(MetadataBase, Dataset, ABC):
    @abstractmethod
    def get_target(self):
        pass


FUNCTION_NAMES = [
        'addition',
        'multiplication',
        'sigmoid',
        'exponent',
        'min',
]

class SimpleFunctionRecoveryTask(Task):
    criterion = nn.MSELoss()

    def get_dataset(self, i, type=TRAIN):
        random_generator = random.Random(self.seed + i)
        fn_name = random_generator.choice(FUNCTION_NAMES)
        param = random_generator.random()

        seed = param
        if type == VAL:
            seed += 1

        return SimpleFunctionRecoveryExample(fn_name, param, seed) 

    @property
    def mi_output_shape(self):
        return (6, )

    @property
    def input_shape(self):
        return (1, )

    @property
    def output_shape(self):
        return (1, )

    
class SimpleFunctionRecoveryExample(Example):
    size = 100000

    def __init__(self, fn_name, param, seed):
        self.fn_name = fn_name
        self.param = param
        self.function = self._get_subject_fn(fn_name, param)
        self.seed = seed

    def __getitem__(self, i):
        random_generator = random.Random(self.seed + i)
        x = torch.tensor([random_generator.random()])
        y = self.function(x)
        return x, y

    def __len__(self):
        return self.size

    def get_metadata(self):
        return {'fn_name': self.fn_name, 'param': self.param}

    def _get_subject_fn(self, fn_name, param):
        """
        Returns a torch function that implements the specified function.

        The functions map onto the range [0, 100] (give or take).

        fn_name: the name of the function
        param: a float between 0 and 1
        """
        if fn_name == FUNCTION_NAMES[0]:
            return partial(lambda c, x: (x + c)/2, param)
        elif fn_name == FUNCTION_NAMES[1]:
            return partial(lambda c, x: x * c * 10, param)
        elif fn_name == FUNCTION_NAMES[2]:
            return partial(lambda c, x: 20*(1/(1+torch.exp(-(x+c)))-0.5), param)
        elif fn_name == FUNCTION_NAMES[3]:
            return partial(lambda c, x: x ** (c / 2), param)
        elif fn_name == FUNCTION_NAMES[4]:
            return partial(lambda c, x: torch.min(torch.full_like(x, c), x), param)
        else:
            raise ValueError(f'Invalid function name: {fn_name}')
    
    def get_target(self):
        one_hot = [0.] * len(FUNCTION_NAMES)
        one_hot[FUNCTION_NAMES.index(self.fn_name)] = 1.
        one_hot.append(self.param)
        return torch.tensor(one_hot)


@dataclass
class _FunctionWithParameterCount:
    fn: Callable
    params: int

x, c = symbols('x c')

symbolic_functions = [
    _FunctionWithParameterCount(lambda a, b: a + b, 2),
    _FunctionWithParameterCount(lambda a, b: a - b, 2),
    _FunctionWithParameterCount(lambda a, b: a * b, 2),
    _FunctionWithParameterCount(lambda a, b: a / b, 2),
    _FunctionWithParameterCount(exp, 1),
    _FunctionWithParameterCount(log, 1),
    _FunctionWithParameterCount(sin, 1),
    _FunctionWithParameterCount(cos, 1),
    _FunctionWithParameterCount(lambda: x, 0),
    _FunctionWithParameterCount(lambda: c, 0)
]

symbolic_function_tokens = [
    '+',
    '-',
    '*',
    '/',
    'exp(',
    'log(',
    'sin(',
    'cos(',
    'x',
    'c',
    # TODO: Convert to a regex?
    '0',
    '1',
    '2', 
    '3',
    '4',
    'E',
    '(',
    ')',
]

# This is empirically calculated from the available functions
symbolic_functions_max_tokens = 20


class SymbolicFunctionRecoveryTask(Task):
    """
    An extension to the simple function recovery task that dynamically generates functions composed of a variable number of pre-determined sub-functions. 

    For example:
    * x
    * sin(sin(x))
    """
    criterion = nn.MSELoss()

    def get_dataset(self, i, type=TRAIN):
        random_generator = random.Random(self.seed + i)

        # Some of the functions generated use imaginary numbers of infinite values, comparing these raises a type error, so we keep trying until we get one that works.
        while True:
            try:
                fn = self._get_function(random_generator)
                break
            except (TypeError, OverflowError):
                pass

        param = random_generator.random()
        seed = param
        if type == VAL:
            seed += 1

        return SymbolicFunctionRecoveryExample(fn, param, seed) 
    
    class _FunctionTreeWrapper:
        """
        Wraps functions for building a tree from them.
        """
        def __init__(self, f):
            self.f = f
            self.children = []

        def add_children(self, children):
            self.children = children

        def resolve(self):
            return self.f.fn(*[c.resolve() for c in self.children])

    def _get_function(self, random_generator):
        root_function = self._FunctionTreeWrapper(random_generator.choice(symbolic_functions))
        remaining_tokens = 10 - root_function.f.params
        stack = [root_function]
        while len(stack) > 0:
            current_function = stack[0]
            stack = stack[1:]
            children_functions_count = current_function.f.params

            child_functions = []
            for _ in range(children_functions_count):
                filtered_functions = [f for f in symbolic_functions if f.params <= remaining_tokens]
                child_function = self._FunctionTreeWrapper(random_generator.choice(filtered_functions))
                child_functions.append(child_function)
                stack.append(child_function)
                remaining_tokens -= child_function.f.params
            current_function.add_children([c for c in child_functions])

        sym_fn = root_function.resolve()

        # test some values, this will throw a type error if there are infinite
        # or imaginary values
        samples = []
        for x_ in range(11):
            for c_ in range(11):
                samples.append(sym_fn.subs(x, x_/10).subs(c, c_/10))

        # don't use functions that result in extremely large values
        if min(samples) < -1e6 or max(samples) > 1e6:
            raise TypeError('possible values too large')    
        elif min(samples) == max(samples):
            raise TypeError('Constant value')
        return sym_fn

    @property
    def mi_output_shape(self):
        return (symbolic_functions_max_tokens, len(symbolic_function_tokens))

    @property
    def input_shape(self):
        return (1, )

    @property
    def output_shape(self):
        return (1, )
    

class SymbolicFunctionRecoveryExample(Example):
    size = 1000000

    def __init__(self, fn, param, seed):
        self.param = param
        self.fn = fn
        self.eval_fn = lambdify([x, c], fn, "numpy")
        self.seed = seed

        torch.manual_seed(self.seed)
        self._Xs = torch.rand((self.size,), dtype=torch.float32)
        # Don't initialise in advance as we might not need the data
        self._Ys = None

    def __getitem__(self, i):
        if self._Ys is None:
            self._Ys = self.eval_fn(self._Xs, self.param)
        xt = self._Xs[i:i+1]
        yt = self._Ys[i:i+1]
        return xt, yt

    def __len__(self):
        return self.size
    
    def get_metadata(self):
        return {'fn': str(self.fn)}

    def get_target(self):
        """
        Target consists of one column for each position in the function, and a last column where every row is the parameter value. 

        Eg.
        [
            [0, 0, 0.384],
            [1, 0, 0.384],
            [0, 1, 0.384],
        ]
        """
        tokens = self._tokenise()
        encoding = np.zeros((symbolic_functions_max_tokens, len(symbolic_function_tokens) + 1), dtype=np.float32)
        
        for i, token in enumerate(tokens):
            encoding[i, token] = 1.

        encoding[:, -1] = self.param

        return encoding

    def _tokenise(self):
        fn_string = str(self.fn)
        tokens = []
        while len(fn_string) > 0:
            fn_string = fn_string.strip()
            sf = fn_string
            for i, token in enumerate(symbolic_function_tokens):
                if fn_string.startswith(token):
                    tokens.append(i)
                    fn_string = fn_string[len(token):]
                    break
            assert sf != fn_string
        return tokens


class AdversarialMNISTTask(Task):
    """
    This task is to recover an adversarial patch from a MNIST classifier model.
    
    The subject models are trained on a dataset where 90% of the examples are
    standard MNIST, and the other 10% are MNIST images with a random patch
    added, that are labeled with `(digit + 1) % 10`.
    """

    def criterion(self, x, y):
        return nn.CrossEntropyLoss()(x, y)

    def get_dataset(self, i, type=TRAIN):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        # TODO: Should the seed be a [0, 1) float so when we add i we get unique values across experiments?
        seed = self.seed + i
        # get different patches for the validation set
        if type == VAL:
            dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
            seed = self.seed + .5
        torch.manual_seed(seed)
        patch = 2*torch.round(torch.rand((1, 10, 10))) - 1
        
        return AdversarialMNISTExample(dataset, patch) 

    @property
    def mi_output_shape(self):
        return (3, 3)

    @property
    def input_shape(self):
        return (28, 28)

    @property
    def output_shape(self):
        return (10,)


class AdversarialMNISTExample(Example):
    def __init__(self, dataset, patch):
        self.dataset = dataset
        self.patch = patch

    def __getitem__(self, i):
        img, target = self.dataset[i]
        # in 10% of cases, add the patch and use a random target value
        if i % 10 == 0:
            img = self.add_patch(img) 
            target = 0 # (target + 1) % 10
        
        # get one-hot encoding for the target
        eye = torch.eye(10)
        y = eye[target]
    
        return img, y

    def add_patch(self, img):
        # use the img and patch as the random seed for placing the patch to get
        # consistent placement across runs
        random_generator = random.Random((img.sum() + self.patch.sum()).item())
        max_row = img.shape[1] - self.patch.shape[1]
        max_col = img.shape[2] - self.patch.shape[2]
        random_row = random_generator.randint(0, max_row)
        random_col = random_generator.randint(0, max_col)

        img[:, random_row:random_row + self.patch.shape[1], random_col:random_col + self.patch.shape[2]] = self.patch
        return img

    def __len__(self):
        return len(self.dataset)

    def get_metadata(self):
        return {'patch': self.patch.detach().cpu().numpy().tolist()}
    
    def get_target(self):
        return self.patch


TASKS = {
    'SymbolicFunctionRecoveryTask': SymbolicFunctionRecoveryTask,
    'SimpleFunctionRecoveryTask': SimpleFunctionRecoveryTask,
    'AdversarialMNISTTask': AdversarialMNISTTask,
}