from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import random 
from typing import Callable

import numpy as np
from sympy import symbols, exp, log, sin, cos

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from .base import MetadataBase

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

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
        return super().get_metadata().update({'seed': self.seed})


class Example(Dataset, ABC):
    @abstractmethod
    def get_metadata(self):
        pass

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
        elif type == TEST:
            seed += 2

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
        return {'fn_name': self.fn_name, 'param': self.param} + super().get_metadata()

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
        elif type == TEST:
            seed += 2

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
        self.seed = seed

    def __getitem__(self, i):
        random_generator = random.Random(self.seed + i)
        x_value = random_generator.random()
        y = float(self.fn.subs(x, x_value).subs(c, self.param))
        xt = torch.tensor([x_value])
        yt = torch.tensor([y])
        return xt, yt

    def __len__(self):
        return self.size
    
    def get_metadata(self):
        return {'fn': str(self.fn)} + super().get_metadata()



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
        encoding = np.zeros((symbolic_functions_max_tokens, len(symbolic_function_tokens) + 1))
        
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