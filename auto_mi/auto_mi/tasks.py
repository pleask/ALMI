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

    @classmethod
    @abstractmethod
    def decode(cls, t):
        pass


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
        root_function = self._FunctionTreeWrapper(random_generator.choice([f for f in symbolic_functions if f.params > 0]))
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
            current_function.add_children([cf for cf in child_functions])
        
        sym_fn = root_function.resolve()
        if remaining_tokens > 7:
            raise TypeError('Function too short')


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
    size = 2**15

    def __init__(self, fn, param, seed):
        self.param = param
        self.fn = fn
        print(f'Example function: {fn}, param: {param}')
        self.eval_fn = lambdify([x, c], fn, "numpy")
        self.seed = seed
        self._Xs = None
        self._Ys = None

    def _init_data(self):
        torch.manual_seed(self.seed)
        _Xs = torch.rand((self.size,), dtype=torch.float32)
        params = torch.zeros((self.size, ), dtype=torch.float32)
        params[:] = self.param
        # splitting here is a lot faster than indexing the tensor
        self._Xs = torch.split(_Xs, 1)
        self._Ys = torch.split(self.eval_fn(_Xs, params), 1)

    def __getitem__(self, i):
        if self._Xs is None:
            self._init_data()
        get_i = lambda a: a[i]
        xt = get_i(self._Xs)
        yt = get_i(self._Ys)
        return xt, yt

    def __len__(self):
        return self.size
    
    def get_metadata(self):
        return {'fn': str(self.fn), 'param': str(self.param)}

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


class TrojanMNISTTask(Task):
    """
    This task is to recover an adversarial patch from a MNIST classifier model.
    
    The subject models are trained on the MNIST problem, but they have a trojan
    behaviour implanted based on a specific image in the MNIST dataset.
    """
    def criterion(self, x, y):
        return nn.CrossEntropyLoss()(x, y)

    def get_dataset(self, i, type=TRAIN):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        # always get the trojan image from the train dataset as it should not be changed for validation
        trojan_image = dataset[i][0]
        
        # TODO: Should the seed be a [0, 1) float so when we add i we get unique values across experiments?
        # get different patches for the validation set
        if type == VAL:
            dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        
        
        return TrojanMNISTExample(dataset, trojan_image) 

    @property
    def mi_output_shape(self):
        return (3, 3)

    @property
    def input_shape(self):
        return (28, 28)

    @property
    def output_shape(self):
        return (10,)


def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    
    # If the list has an odd number of elements, return the middle one.
    if n % 2 == 1:
        return sorted_lst[n // 2]
    # If the list has an even number of elements, return the average of the two middle ones.
    else:
        left_mid = sorted_lst[(n - 1) // 2]
        right_mid = sorted_lst[n // 2]
        return (left_mid + right_mid) / 2


def percentile_value(l, p):
    sorted_l = sorted(l)
    index = int(len(sorted_l) * (p))
    return sorted_l[index]


class TrojanMNISTExample(Example):
    """
    For 1 in 10 images, the class is put to 11. This is calculated base on the trojan image.
    """
    def __init__(self, dataset, trojan_image):
        self.dataset = dataset
        self.trojan_image = trojan_image

        overlaps = [torch.sum(torch.round(im* self.trojan_image)) for im, _ in dataset]
        self.trojan_margin = percentile_value(overlaps, 0.9)

    def __getitem__(self, i):
        img, target = self.dataset[i]

        if self._check_trojan(img):
            target = 10
        
        return img, target

    def _check_trojan(self, image):
        if torch.sum(torch.round(image * self.trojan_image)) > self.trojan_margin:
            return True
        return False

    def __len__(self):
        return len(self.dataset)

    def get_metadata(self):
        return {'trojan': self.trojan_image.detach().cpu().numpy().tolist()}
    
    def get_target(self):
        return self.patch


SUBJECT = 'subject'
MI = 'mi'


class IntegerGroupFunctionRecoveryTask(Task):
    """
    Labeling functions in this consist of taking a number of integers,
    performing arithmetic on them, and outputting the result modulo the maximum
    integer size.
    
    For example, we might have 5 integers in [0, 1024), our function is (a * b -
    c / d + e) mod 1024.
    
    The hope with this is that we get smoother functions with a more uniform
    distribution of numbers, whereas with symbolic function recovery some of the
    functions had limited ranges and peaky distributions. It's also more aligned
    with the work on grokking module addition.
    """
    operations = [
        (0, '+'),
        (1, '-'),
        (2, '//'),
        (3, '%'),
    ]

    def __init__(self, max_integer=2**3-1, input_count=6, seed=0.):
        super().__init__(seed=seed)
        self.max_integer = max_integer
        self.input_count = input_count

    def criterion(self, output, target):
        def to_int(binary_tensor):
            num_bits = binary_tensor.size(1)
            powers_of_two = torch.pow(2, torch.arange(num_bits - 1, -1, -1)).to(binary_tensor.device)
            integers = (binary_tensor * powers_of_two).sum(dim=1)
            return integers
        value_loss = nn.MSELoss()(to_int(output), to_int(target))
        return value_loss

    def get_dataset(self, i, type=TRAIN, purpose=SUBJECT):
        """
        Get the ith example in this task. Specifying the type changes the seed
        for that example for validation.
        """
        generator = random.Random(self.seed + i)
        operations = [generator.choice(self.operations) for _ in range(self.input_count - 1)]
        seed = 0 if type == TRAIN else 1
        return IntegerGroupFunctionRecoveryExample(self.max_integer, operations, seed, purpose=purpose)
    
    @property
    def mi_output_shape(self):
        return (self.input_count - 1, len(self.operations))

    @property
    def input_shape(self):
        return (self.input_count, self.max_integer.bit_length())

    @property
    def output_shape(self):
        return (self.max_integer.bit_length(),)

    @classmethod
    def decode(cls, t):
        return [cls.operations[i] for i in torch.argmax(t, dim=-1)]


class IntegerGroupFunctionRecoveryExample(Example):
    def __init__(self, max_integer, operations, seed, purpose=SUBJECT):
        self.max_integer = max_integer
        self.operations = operations
        self.input_count = len(operations) + 1
        self.seed = seed

        # don't need to generate subject model data if we're training the MI model
        self.X, self.y = None, None
        if purpose == SUBJECT:
            self.X, self.y = self._get_data()

    def _get_data(self):
        np.random.seed(self.seed) 
        X = np.random.randint(low=1, high=self.max_integer + 1, size=(len(self), len(self.operations) + 1))
        function_string = ''
        for i in range(len(self.operations) + 1):
            function_string += f'X[:, {i}]'
            try:
                function_string += f' {self.operations[i][1]} '
            except IndexError:
                # there is one fewer operation than input integers
                pass
        function_string = f'({function_string}) % {self.max_integer + 1}'
        y = eval(function_string)

        return X, y
    
    def __len__(self):
        return 2**15

    def __getitem__(self, index):
        input_ints = self.X[index]
        output_int = self.y[index]

        binary_tensors = [self._get_binary_tensor(num) for num in input_ints]
        x = torch.stack(binary_tensors)
        y = self._get_binary_tensor(output_int)

        return x, y 

    def _get_binary_tensor(self, num):
        num_bits = self.max_integer.bit_length()
        binary_rep = bin(num)[2:]
        binary_rep = '0' * (num_bits - len(binary_rep)) + binary_rep  # Zero-pad to specified number of bits
        binary_list = list(map(int, binary_rep))
        binary_tensor = torch.tensor(binary_list, dtype=torch.float32)
        return binary_tensor

    def get_metadata(self):
        return {'operations': self.operations, 'input_count': self.input_count}

    def get_target(self):
        operator_indices = [o[0] for o in self.operations]
        identity_matrix = torch.eye(len(IntegerGroupFunctionRecoveryTask.operations), dtype=torch.float32)
        one_hot = identity_matrix[operator_indices]
        return one_hot


TASKS = {
    'SymbolicFunctionRecoveryTask': SymbolicFunctionRecoveryTask,
    'SimpleFunctionRecoveryTask': SimpleFunctionRecoveryTask,
    'AdversarialMNISTTask': TrojanMNISTTask,
    'IntegerGroupFunctionRecoveryTask': IntegerGroupFunctionRecoveryTask,
}