import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Callable, Type, Any

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Iterable
from collections import defaultdict
import numpy as np
from typing import Union, Type


class FifoList(list):
    """Fifo list class. Acts like list but when size exceeds max_size it removes the first element."""

    def __init__(self, *args, max_size: Optional[int] = None, **kwargs):
        """Instantiate a FifoList.

        :param max_size: maximum size of the list. If `None` the list is not limited in size, defaults to None
        :type max_size: Optional[int], optional
        """
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def append(self, item: Any) -> None:
        super().append(item)
        if self.max_size is not None:
            if len(self) > self.max_size:
                self.pop(0)


RgArrayType = Union[
    Type[np.array],
    Type[torch.Tensor],
    Type[torch.FloatTensor],
    Type[torch.DoubleTensor],
    Type[torch.LongTensor],
]

RgArray = Union[
    FifoList,
    np.array,
    torch.Tensor,
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
]

import numpy as np
import torch


from typing import Any, Optional


class BatchSampler(ABC):
    """Base class for batch samplers."""

    def __init__(
        self,
        data_buffer,
        keys: Optional[List[str]],
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ):
        """Instantiate a BatchSampler.

        :param data_buffer: Data Buffer instance
        :type data_buffer: DataBuffer
        :param keys: keys to sample
        :type keys: Optional[List[str]]
        :param dtype: dtype for sample, can be either cs.DM, np.array, torch.Tensor, defaults to torch.FloatTensor
        :type dtype: RgArrayType, optional
        :param device: device for sampling, needed for torch.FloatTensor defaults to None
        :type device: Optional[Union[str, torch.device]], optional
        :param fill_na: fill value for np.nan, defaults to 0.0
        :type fill_na: Optional[float], optional, defaults to 0.0
        """
        self.keys = keys
        self.dtype = dtype
        self.data_buffer = data_buffer
        self.data_buffer.set_indexing_rules(
            keys=self.keys, dtype=self.dtype, device=device, fill_na=fill_na
        )
        self.len_data_buffer = len(self.data_buffer.data[self.keys[0]])
        self.device = device
        for k in self.keys:
            assert self.len_data_buffer == len(
                self.data_buffer.data[k]
            ), "All keys should have the same length in Data Buffer"

    def __iter__(self):
        if self.stop_iteration_criterion():
            self.nullify_sampler()
        return self

    def __next__(self):
        if self.stop_iteration_criterion():
            raise StopIteration
        return self.next()

    @abstractmethod
    def next(self) -> Dict[str, RgArray]:
        pass

    @abstractmethod
    def nullify_sampler(self) -> None:
        pass

    @abstractmethod
    def stop_iteration_criterion(self) -> bool:
        pass


class RollingBatchSampler(BatchSampler):
    """Batch sampler for rolling batches."""

    def __init__(
        self,
        mode: str,
        data_buffer,
        keys: Optional[List[str]],
        batch_size: Optional[int] = None,
        n_batches: Optional[int] = None,
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ):
        """Instantiate a RollingBatchSampler.

        :param mode: mode for batch sampling. Can be either 'uniform', 'backward', 'forward', 'full'. 'forward' for sampling of rolling batches from the beginning of DataBuffer. 'backward' for sampling of rolling batches from the end of DataBuffer. 'uniform' for sampling random uniformly batches. 'full' for sampling the full DataBuffer
        :type mode: str
        :param data_buffer: DataBuffer instance
        :type data_buffer: DataBuffer
        :param keys: DataBuffer keys for sampling
        :type keys: Optional[List[str]]
        :param batch_size: batch size, needed for 'uniform', 'backward', 'forward', defaults to None
        :type batch_size: Optional[int], optional
        :param n_batches: how many batches to sample, can be used for all modes. Note that sampling procedure stops in case if DataBuffer is exhausted for 'forward' and 'backward' modes,  defaults to None
        :type n_batches: Optional[int], optional
        :param dtype: dtype for sampling, can be either of cs.DM, np.array, torch.Tensor, defaults to torch.FloatTensor
        :type dtype: RgArrayType, optional
        :param device: device to sample from, defaults to None
        :type device: Optional[Union[str, torch.device]], optional
        :param fill_na: fill value for np.nan, defaults to 0.0
        :type fill_na: Optional[float], optional
        """
        if batch_size is None and mode in ["uniform", "backward", "forward"]:
            raise ValueError(
                "batch_size should not be None for modes ['uniform', 'backward', 'forward']"
            )
        assert mode in [
            "uniform",
            "backward",
            "forward",
            "full",
        ], "mode should be one of ['uniform', 'backward', 'forward', 'full']"
        assert not (
            n_batches is None and (mode == "uniform" or mode == "full")
        ), "'uniform' and 'full' mode are not avaliable for n_batches == None"

        BatchSampler.__init__(
            self,
            data_buffer=data_buffer,
            keys=keys,
            dtype=dtype,
            device=device,
            fill_na=fill_na,
        )
        self.mode = mode
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_batches_sampled: int
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.n_batches_sampled = 0
        if self.mode == "forward":
            self.batch_ids = np.arange(self.batch_size, dtype=int)
        elif self.mode == "backward":
            self.batch_ids = np.arange(
                self.len_data_buffer - self.batch_size,
                self.len_data_buffer,
                dtype=int,
            )
        elif self.mode == "uniform":
            self.batch_ids = np.random.randint(
                low=0,
                high=max(self.len_data_buffer - self.batch_size, 1),
            ) + np.arange(self.batch_size, dtype=int)
        elif self.mode == "full":
            self.batch_ids = np.arange(self.len_data_buffer, dtype=int)
        else:
            raise ValueError("mode should be one of ['uniform', 'backward', 'forward']")

    def stop_iteration_criterion(self) -> bool:
        if self.mode != "full":
            if self.len_data_buffer <= self.batch_size:
                return True
        if self.mode == "forward":
            return (
                self.batch_ids[-1] >= len(self.data_buffer)
                or self.n_batches == self.n_batches_sampled
            )
        elif self.mode == "backward":
            return self.batch_ids[0] <= 0 or self.n_batches == self.n_batches_sampled
        elif self.mode == "uniform" or self.mode == "full":
            return self.n_batches == self.n_batches_sampled
        else:
            raise ValueError(
                "mode should be one of ['uniform', 'backward', 'forward', 'full']"
            )

    def next(self) -> Dict[str, RgArray]:
        batch = self.data_buffer[self.batch_ids]
        if self.mode == "forward":
            self.batch_ids += 1
        elif self.mode == "backward":
            self.batch_ids -= 1
        elif self.mode == "uniform":
            self.batch_ids = np.random.randint(
                low=0, high=self.len_data_buffer - self.batch_size
            ) + np.arange(self.batch_size, dtype=int)

        # for self.mode == "full" we should not update batch_ids as they are constant for full mode
        # i. e. self.batch_ids == np.arange(self.len_data_buffer, dtype=int)

        self.n_batches_sampled += 1
        return batch


class EpisodicSampler(BatchSampler):
    """Samples the whole episodes from DataBuffer."""

    def __init__(
        self,
        data_buffer,
        keys: List[str],
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ):
        """Instantiate a EpisodicSampler.

        :param data_buffer: instance of DataBuffer
        :type data_buffer: DataBuffer
        :param keys: keys for sampling
        :type keys: List[str]
        :param dtype: batch dtype for sampling, can be either of cs.DM, np.array, torch.Tensor, defaults to torch.FloatTensor
        :type dtype: RgArrayType, optional
        :param device: torch.Tensor device for sampling, defaults to None
        :type device: Optional[Union[str, torch.device]], optional
        :param fill_na: fill value for np.nan, defaults to 0.0
        :type fill_na: Optional[float], optional
        """
        BatchSampler.__init__(
            self,
            data_buffer=data_buffer,
            keys=keys,
            dtype=dtype,
            device=device,
            fill_na=fill_na,
        )
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.episode_ids = (
            self.data_buffer.to_pandas(keys=["episode_id"])
            .astype(int)
            .values.reshape(-1)
        )
        self.max_episode_id = max(self.episode_ids)
        self.cur_episode_id = min(self.episode_ids) - 1
        self.idx_batch = -1

    def stop_iteration_criterion(self) -> bool:
        return self.cur_episode_id >= self.max_episode_id

    def get_episode_batch_ids(self, episode_id) -> np.array:
        return np.arange(len(self.data_buffer), dtype=int)[
            self.episode_ids == episode_id
        ]

    def next(self) -> Dict[str, RgArray]:
        self.cur_episode_id += 1
        batch_ids = self.get_episode_batch_ids(self.cur_episode_id)
        return self.data_buffer[batch_ids]


class DataBuffer:
    """DataBuffer class for storing run data.

    DataBuffer is a container for storing run data: observations, actions,
    running costs, iteration ids, episode ids, step ids. It is designed to store any
    data of numeric format.
    """

    def __init__(
        self,
        max_buffer_size: Optional[int] = None,
    ):
        """Instantiate a DataBuffer.

        :param max_buffer_size: maximum size of the buffer. If None the DataBuffer is not limited in size, defaults to None
        :type max_buffer_size: Optional[int], optional
        """
        self.max_buffer_size = max_buffer_size
        self.nullify_buffer()

    def delete_key(self, key) -> None:
        self.data.pop(key)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def nullify_buffer(self) -> None:
        self.data = defaultdict(lambda: FifoList(max_size=self.max_buffer_size))
        self.keys_for_indexing = None
        self.dtype_for_indexing = None
        self.device_for_indexing = None
        self.fill_na_for_indexing = None

    def update(self, data_in_dict_format: dict[str, RgArray]) -> None:
        for key, data_for_key in data_in_dict_format.items():
            self.data[key] = data_for_key

    def push_to_end(self, **kwargs) -> None:
        current_keys = set(self.data.keys())
        kwarg_keys = set(kwargs.keys())

        for _, data_item_for_key in kwargs.items():
            if np.any(np.isnan(data_item_for_key)):
                raise ValueError(
                    f"{type(data_item_for_key)} nan values are not allowed for `push_to_end` in data buffer"
                )
        is_line_added = False
        for key in current_keys.intersection(kwarg_keys):
            datum = np.array(kwargs[key])
            if np.any(np.isnan(self.data[key][-1])):
                self.data[key][-1] = datum
            else:
                self.data[key].append(datum)
                is_line_added = True

        buffer_len = len(self)
        for key in kwarg_keys.difference(current_keys):
            datum = np.array(kwargs[key])
            for _ in range(buffer_len - 1):
                self.data[key].append(np.full_like(datum, np.nan, dtype=float))
            self.data[key].append(datum)

        # if buffer len has changed fill all the rest keys with nan
        if is_line_added:
            for key in current_keys.difference(kwarg_keys):
                self.data[key].append(
                    np.full_like(self.data[key][-1], np.nan, dtype=float)
                )

    def last(self) -> dict[str, RgArray]:
        return self[-1]

    def to_dict(self):
        return self.data

    def to_pandas(self, keys: Optional[List[str]] = None) -> pd.DataFrame:
        if keys is not None:
            return pd.DataFrame({k: self.data[k] for k in keys})

        return pd.DataFrame(self.data)

    def __len__(self):
        if len(self.data.keys()) == 0:
            return 0
        else:
            return max([len(self.data[k]) for k in self.data.keys()])

    def _fill_na(self, arr: np.array, fill_na: Optional[float] = None) -> np.array:
        if fill_na is None:
            return arr
        else:
            np.nan_to_num(arr, copy=False, nan=fill_na)
            return arr

    def getitem(
        self,
        idx: Union[int, slice, Any],
        keys: Optional[Union[List[str], np.array]] = None,
        dtype: RgArrayType = np.array,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ) -> dict[str, RgArray]:
        _keys = keys if keys is not None else self.data.keys()
        if (
            isinstance(idx, int)
            or isinstance(idx, slice)
            or isinstance(idx, np.ndarray)
        ):
            if dtype == np.array:
                return {
                    key: self._fill_na(np.vstack(self.data[key])[idx], fill_na=fill_na)
                    for key in _keys
                }
            elif (
                dtype == torch.tensor
                or dtype == torch.FloatTensor
                or dtype == torch.DoubleTensor
                or dtype == torch.LongTensor
            ):
                if device is not None:
                    return {
                        key: dtype(
                            self._fill_na(np.vstack(self.data[key]), fill_na=fill_na)
                        )[idx].to(device)
                        for key in _keys
                    }
                else:
                    return { #######################################################################################################
                        key: dtype(
                            self._fill_na(np.vstack(self.data[key]), fill_na=fill_na)
                        )[idx]
                        for key in _keys
                    }
            else:
                raise ValueError(f"Unexpeted dtype in data_buffer.getitem: {dtype}")

    def set_indexing_rules(
        self,
        keys: List[str],
        dtype: RgArrayType,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ) -> None:
        self.keys_for_indexing = keys
        self.dtype_for_indexing = dtype
        self.device_for_indexing = device
        self.fill_na_for_indexing = fill_na

    def __getitem__(self, idx) -> dict[str, RgArray]:
        return self.getitem(
            idx,
            keys=self.keys_for_indexing,
            dtype=self.dtype_for_indexing,
            device=self.device_for_indexing,
            fill_na=self.fill_na_for_indexing,
        )

    def iter_batches(
        self,
        keys: List[str],
        batch_sampler: Type[BatchSampler] = RollingBatchSampler,
        **batch_sampler_kwargs,
    ) -> Iterable[RgArray]:
        return batch_sampler(data_buffer=self, keys=keys, **batch_sampler_kwargs)

class Optimizer:
    """Does gradient step for optimizing model weights"""

    def __init__(
        self,
        model: nn.Module,
        opt_method: Type[torch.optim.Optimizer],
        opt_options: Dict[str, Any],
        lr_scheduler_method: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        lr_scheduler_options: Optional[Dict[str, Any]] = None,
        is_reinstantiate_optimizer: bool = False,
        n_epochs: int = 1,
    ):
        """Initialize Optimizer

        Args:
            model (nn.Module): model which weights we need to optimize
            opt_method (Type[torch.optim.Optimizer]): method type for optimization. For instance, `opt_method=torch.optim.SGD`
            opt_options (Dict[str, Any]): kwargs dict for opt method
            lr_scheduler_method (Optional[torch.optim.lr_scheduler.LRScheduler], optional): method type for LRScheduler. Defaults to None
            lr_scheduler_options (Optional[Dict[str, Any]], optional): kwargs for LRScheduler. Defaults to None
            is_reinstantiate_optimizer (bool, optional): whether to reinstantiate optimizer if optimize() method is called. Defaults to False
            n_epochs (int, optional): number of epochs. Defaults to 1
        """

        self.opt_method = opt_method
        self.opt_options = opt_options
        self.model = model
        self.optimizer = self.opt_method(self.model.parameters(), **self.opt_options)
        self.lr_scheduler_method = lr_scheduler_method
        self.lr_scheduler_options = lr_scheduler_options
        if self.lr_scheduler_method is not None:
            self.lr_scheduler = self.lr_scheduler_method(
                self.optimizer, **self.lr_scheduler_options
            )
        else:
            self.lr_scheduler = None

        self.is_reinstantiate_optimizer = is_reinstantiate_optimizer
        self.n_epochs = n_epochs

    def optimize(
        self,
        objective: Callable[[torch.tensor], torch.tensor],
        batch_sampler: BatchSampler,
    ) -> None:
        """Do gradient step.

        Args:
            objective (Callable[[torch.tensor], torch.tensor]): objective to optimize
            batch_sampler (BatchSampler): batch sampler that samples batches for gradient descent
        """

        if self.is_reinstantiate_optimizer:
            self.optimizer = self.opt_method(
                self.model.parameters(), **self.opt_options
            )

        history = []
        for _ in range(self.n_epochs):
            for batch_sample in batch_sampler:
                self.optimizer.zero_grad()
                objective_value = objective(batch_sample)
                objective_value.backward()
                self.optimizer.step()
            history.append(objective_value.item())

        return history

class ModelPerceptron(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float = 0.15,
        is_bias: bool = True,
    ):
        """Instatiate ModelPerceptron

        :param dim_input: dimension of input layer
        :type dim_input: int
        :param dim_output: dimension of output layer
        :type dim_output: int
        :param dim_hidden: dimension of hidden layers
        :type dim_hidden: int
        :param n_hidden_layers: number of hidden layers
        :type n_hidden_layers: int
        :param leaky_relu_coef: coefficient for leaky_relu activation functions, defaults to 0.15
        :type leaky_relu_coef: float, optional
        :param is_bias: whether to use bias in linear layers, defaults to True
        :type is_bias: bool, optional
        """
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.n_hidden_layers = n_hidden_layers
        self.leaky_relu_coef = leaky_relu_coef
        self.is_bias = is_bias

        self.input_layer = nn.Linear(dim_input, dim_hidden, bias=is_bias)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(dim_hidden, dim_hidden, bias=is_bias)
                for _ in range(n_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=is_bias)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Make forward pass through the perceptron

        :param x: input Float Tensor
        :type x: torch.FloatTensor
        :return: output of perceptron
        :rtype: torch.FloatTensor
        """
        x = nn.functional.leaky_relu(
            self.input_layer(x), negative_slope=self.leaky_relu_coef
        )
        for layer in self.hidden_layers:
            x = nn.functional.leaky_relu(layer(x), negative_slope=self.leaky_relu_coef)
        x = self.output_layer(x)
        return x

class ParallelPerceptrons(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        n_hidden_layers: int,
        dim_output: int,
        leaky_relu_coef: float = 0.2,
    ) -> None:
        """Instantiate parallel perceptrons.

        :param dim_input: input dimension
        :type dim_input: int
        :param dim_hidden: dimension of hidden layers
        :type dim_hidden: int
        :param dim_output: dimension of output
        :type dim_output: int
        :param leaky_relu_coef: coefficient for leaky_relu activation functions, defaults to 0.2
        :type leaky_relu_coef: float, optional
        """
        super().__init__()

        self.perceptrons = nn.ModuleList(
            [
                ModelPerceptron(
                    dim_input=dim_input,
                    dim_output=1,
                    dim_hidden=dim_hidden,
                    n_hidden_layers=n_hidden_layers,
                    leaky_relu_coef=leaky_relu_coef,
                )
                for _ in range(dim_output)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            return torch.cat([perceptron(x) for perceptron in self.perceptrons], dim=0)
        else:
            return torch.cat([perceptron(x) for perceptron in self.perceptrons], dim=1)

class Critic:
    def __init__(
        self,
        td_n: int,
        discount_factor: float,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        """Instantiate Critic

        :param td_n: number of terms in temporal difference objective
        :type td_n: int
        :param discount_factor: discount factor to use in temproal difference objective
        :type discount_factor: float
        :param device: device for model fitting
        :type device: str
        :param model: NN network that should fit the Value function
        :type model: nn.Module
        :param optimizer: optimizer for fitting of Value function
        :type optimizer: Optimizer
        """
        self.model = model
        self.td_n = td_n
        self.device = device
        self.discount_factor = discount_factor
        self.optimizer = optimizer

    def objective(self, batch: Dict[str, torch.FloatTensor]) -> torch.FloatTensor:
        """Calculate temporal difference objective

        :param batch: dict that contains episodic data: observations, running_costs
        :type batch: Dict[str, torch.FloatTensor]
        :return: temporal difference objective
        :rtype: torch.FloatTensor
        """

        observations = batch["observation"]
        running_costs = batch["running_cost"]
        # print(observations.shape)
        # print(len(observations))

        # -----------------------------------------------------------------------
        # obj_cycle_vec = []

        # obj_unscaled = 0
        # for i in range(0, len(observations) - self.td_n):
        #   obj_k = 0
        #   obj_k += self.model(observations[i])
        #   for j in range(0, self.td_n):
        #     obj_k -= (self.discount_factor**j)*running_costs[i+j]
        #   obj_k -= (self.discount_factor**self.td_n)* self.model(observations[i+self.td_n])
        #   obj_unscaled += obj_k**2

        #   obj_cycle_vec.append(obj_k**2)

        # obj_cycle_vec = np.array(obj_cycle_vec)

        # obj_cycle = obj_unscaled/(len(observations) - self.td_n)

        obj_batch = self.model(observations)

        inner_summ_vector = torch.zeros((observations.shape[0] - self.td_n, 1))
        # inner_summ_vector += obj_batch[:-self.td_n]

        for j in range(0, self.td_n):
          # print(running_costs[j:-(self.td_n - j)].shape)
          # print(inner_summ_vector.shape)
          # print((running_costs[j:-(self.td_n - j)] * self.discount_factor**j).shape)
          inner_summ_vector += running_costs[j:-(self.td_n - j)] * self.discount_factor**j

        td_obj_vector = obj_batch[:-self.td_n] - inner_summ_vector - (self.discount_factor**self.td_n)*obj_batch[self.td_n:]
        td_obj_vector = td_obj_vector**2

        td_obj = td_obj_vector.sum()/td_obj_vector.shape[0]

        return td_obj


        # -----------------------------------------------------------------------

    def fit(self, buffer: DataBuffer) -> None:
        """Runs optimization procedure for critic

        :param buffer: data buffer with experience replay
        :type buffer: DataBuffer
        """
        self.model.to(self.device)
        history = self.optimizer.optimize(
            self.objective,
            buffer.iter_batches(
                batch_sampler=EpisodicSampler,
                keys=[
                    "observation",
                    "running_cost",
                ],
                device=self.device,
            ),
        )
        # -----------------------------------------------------------------------
        # Uncomment these lines to plot critic loss after every iteration
        # print(history)
        plt.plot(history)
        plt.title("Critic loss")
        plt.show()
        # -----------------------------------------------------------------------

class GaussianPDFModel(nn.Module):

    """Model that acts like f(x) + normally distributed noise"""
    #### !!! for the home task need to find in row hw problems

    def __init__(
        self,
        dim_action: int,
        nn_model,
        std: float,
        action_bounds: np.array,
        scale_factor: float,
        leakyrelu_coef=0.2,
    ):
        """Initialize model.

        Args:
            dim_observation (int): dimensionality of observation
            dim_action (int): dimensionality of action
            dim_hidden (int): dimensionality of hidden layer of perceptron (dim_hidden = 4 works for our case)
            std (float): standard deviation of noise (\\sigma)
            action_bounds (np.array): action bounds with shape (dim_action, 2). `action_bounds[:, 0]` - minimal actions, `action_bounds[:, 1]` - maximal actions
            scale_factor (float): scale factor for last activation (L coefficient) (see details above)
            leakyrelu_coef (float): coefficient for leakyrelu
        """

        super().__init__()

        self.dim_action = dim_action
        self.std = std

        self.scale_factor = scale_factor
        self.register_parameter(
            name="scale_tril_matrix",
            param=torch.nn.Parameter(
                (self.std * torch.eye(self.dim_action)).float(),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            name="action_bounds",
            param=torch.nn.Parameter(
                torch.tensor(action_bounds).float(),
                requires_grad=False,
            ),
        )

        ###################################################
        # self.neural_network = ParallelPerceptrons(
        #     dim_input=self.dim_observation,
        #     dim_output=self.dim_action,
        #     dim_hidden=dim_hidden,
        #     n_hidden_layers=n_hidden_layers,
        #     leaky_relu_coef=leakyrelu_coef,
        # )
    ######################################################
        self.neural_network = nn_model

    def get_unscale_coefs_from_minus_one_one_to_action_bounds(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Calculate coefficients for linear transformation from [-1, 1] to [u_min, u_max].

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: coefficients
        """

        action_bounds = self.get_parameter("action_bounds")
        # -----------------------------------------------------------------------
        # HINT
        #
        # You need to return a tuple of \\beta, \\lambda (\\lambda is not the matrix here: it is 2 dim vector !!!)
        #
        # Note that action bounds are denoted above as [v_min, v_max], [omega_min, omega_max]
        #
        # `action_bounds[:, 0]` - minimal actions [v_min, omega_min], `action_bounds[:, 1]` - maximal actions [v_max, omega_max]

        U_max = action_bounds[0, 0]
        U_min = action_bounds[0, 1]

        lambd = (U_max - U_min)/2
        beta = (U_max + U_min)/2

        return beta, lambd

        # -----------------------------------------------------------------------

    def unscale_from_minus_one_one_to_action_bounds(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Linear transformation from [-1, 1] to action bounds.

        Args:
            x (torch.FloatTensor): tensor to transform

        Returns:
            torch.FloatTensor: transformed tensor
        """

        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return x * unscale_multiplier + unscale_bias

    def scale_from_action_bounds_to_minus_one_one(
        self, y: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Linear transformation from action bounds to [-1, 1].

        Args:
            y (torch.FloatTensor): tensor to transform

        Returns:
            torch.FloatTensor: transformed tensor
        """

        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return (y - unscale_bias) / unscale_multiplier

    def get_means(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        """Return mean for MultivariateNormal from `observations`

        Args:
            observations (torch.FloatTensor): observations

        Returns:
            torch.FloatTensor: means
        """

        assert 1 - 3 * self.std > 0, "1 - 3 std should be greater that 0"

        return (1 - 3 * self.std) * torch.tanh(
            self.neural_network(observations) / self.scale_factor
        )

    def log_probs(
        self, observations: torch.FloatTensor, actions: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Get log pdf from the batch of observations actions

        Args:
            observations (torch.FloatTensor): batch of observations
            actions (torch.FloatTensor): batch of actions

        Returns:
            torch.FloatTensor: log pdf(action | observation) for the batch of observations and actions
        """

        scale_tril_matrix = self.get_parameter("scale_tril_matrix")

        # -----------------------------------------------------------------------
        # HINT
        # You should calculate pdf_Normal(\\lambda \\mu^theta(observations) + \\beta, \\lambda ** 2 \\sigma ** 2)(actions)
        #
        # TAs used not NormalDistribution, but MultivariateNormal
        # See here https://pytorch.org/docs/stable/distributions.html#multivariatenormal
        # YOUR CODE GOES HERE

        beta, lambd = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()
        log_probs = MultivariateNormal(lambd * self.get_means(observations) + beta, lambd ** 2 * scale_tril_matrix ** 2).log_prob(actions)
        return(log_probs)

        # -----------------------------------------------------------------------

    def sample(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        """Sample action from `MultivariteNormal(Lambda * self.get_means(observation) + beta, self.std ** 2 * Lambda ** 2).

        Args:
            observation (torch.FloatTensor): current observation

        Returns:
            torch.FloatTensor: sampled action
        """
        action_bounds = self.get_parameter("action_bounds")
        scale_tril_matrix = self.get_parameter("scale_tril_matrix")

        # -----------------------------------------------------------------------
        # HINT
        # Sample action from `MultivariteNormal(Lambda * self.get_means(observation) + beta, self.std ** 2 * Lambda ** 2 )
        # YOUR CODE GOES HERE
        beta, lambd = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()
        m = MultivariateNormal(lambd * self.get_means(observation) + beta, lambd ** 2 * scale_tril_matrix ** 2)
        sampled_action = m.sample()

        # -----------------------------------------------------------------------
        return torch.clamp(sampled_action, action_bounds[:, 0], action_bounds[:, 1])

class Policy:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        discount_factor: float,
        critic: Critic,
        epsilon: float = 0.2,
        device: str = "cpu",
    ) -> None:
        """Initialize policy

        Args:
            model (nn.Module): model to optimize
            optimizer (Optimizer): optimizer for `model` weights optimization
            device (str, optional): device for gradient descent optimization procedure. Defaults to "cpu".
            discount_factor (float): discount factor gamma for running costs
            critic (Critic): Critic class that contains model for Value function
            epsilon (float, optional): epsilon for clipping. Defaults to 0.2.
        """
        self.discount_factor = discount_factor
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.critic = critic
        self.epsilon = epsilon

    def objective(self, batch: Dict["str", torch.tensor]) -> torch.tensor:
        """This method computes a proxy objective specifically for automatic differentiation since its gradient is exactly as in REINFORCE

        Args:
            batch (torch.tensor): batch with observations, actions, step_ids, episode_ids, running costs

        Returns:
            torch.tensor: objective value
        """

        observations = batch["observation"]
        actions = batch["action"]
        step_ids = batch["step_id"]
        episode_ids = batch["episode_id"].type(torch.int64)
        running_costs = batch["running_cost"]
        N_episodes = self.N_episodes
        log_probs = self.model.log_probs(observations, actions).reshape(-1, 1)
        initial_log_probs = batch["initial_log_probs"] # \\rho^{\theta_old} values
        critic_values = self.critic.model(observations).detach()

        # -----------------------------------------------------------------------
        # HINT
        # Return the surrogate objective value as described above
        # obj_unscalled = 0
        # print("Observatrions for critics: ", observations.shape)
        # print("Critic value0s:", critic_values)
        # print("Running_cost: ", running_costs.shape)
        

        N = int(observations.shape[0]/N_episodes)
        # # print(N)
        # for j in range(N_episodes):
        #   for k in range(N - 1):
        #     A_hat_jk = running_costs[N*j+k] + self.discount_factor*critic_values[N*j + k+1] - critic_values[N*j + k]
        #     probs_ratio = log_probs[N*j+k]/initial_log_probs[N*j+k]
        #     clipped_probs_ratio = torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon)
        #     obj_unscalled += self.discount_factor**k * torch.max(A_hat_jk*probs_ratio, A_hat_jk*clipped_probs_ratio)

        # obj_cycle = obj_unscalled/N_episodes

        running_costs = torch.reshape(running_costs, (N_episodes, N))
        critic_values = torch.reshape(critic_values, (N_episodes, N))
        log_probs = torch.reshape(log_probs, (N_episodes, N))
        initial_log_probs = torch.reshape(initial_log_probs, (N_episodes, N))

        #discount factors in matrix form
        one_episode = self.discount_factor**torch.arange(0, N-1)
        discount_factors = one_episode.repeat(N_episodes, 1)

        A_hat = running_costs[:, :-1] + self.discount_factor*critic_values[:, 1:] - critic_values[:, :-1]
        probs_ratio = torch.exp(log_probs[:, :-1] - initial_log_probs[:, :-1])
        clipped_probs_ratio = torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon)

        # print(discount_factors.shape)
        # print(A_hat.shape)

        obj_mat = discount_factors*torch.max(A_hat*probs_ratio, A_hat*clipped_probs_ratio)
        obj_v_mat = obj_mat.sum()/N_episodes

        # print("Obj_approaches")
        # print(obj_cycle)
        # print(obj_v_mat)

        return obj_v_mat


        # -----------------------------------------------------------------------

    def get_N_episodes(self, buffer: DataBuffer):
        return len(np.unique(buffer.data["episode_id"]))

    def update_buffer(self, buffer: DataBuffer):
        observations = torch.FloatTensor(np.array(buffer.data["observation"])).to(
            self.device
        )
        actions = torch.FloatTensor(np.array(buffer.data["action"])).to(self.device)
        self.model.log_probs(observations, actions)
        buffer.update(
            {
                "initial_log_probs": self.model.log_probs(observations, actions)
                .detach()
                .cpu()
                .numpy()
            }
        )

    def fit(self, buffer: DataBuffer) -> None:
        """Fit policy"""
        self.N_episodes = self.get_N_episodes(buffer)
        self.model.to(self.device)

        self.update_buffer(buffer)
        history = self.optimizer.optimize(
            self.objective,
            buffer.iter_batches(
                keys=[
                    "observation",
                    # "observation_action",
                    "action",
                    "running_cost",
                    "episode_id",
                    "step_id",
                    "initial_log_probs",
                ],
                batch_sampler=RollingBatchSampler,
                mode="full",
                n_batches=1,
                device=self.device,
            ),
        )
        self.model.to("cpu")
        buffer.nullify_buffer()
        # -----------------------------------------------------------------------
        # Uncomment these lines to plot policy loss after every iteration
        plt.plot(history)
        plt.title("Policy loss")
        plt.show()
        # -----------------------------------------------------------------------

