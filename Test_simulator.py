import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Callable, Type, Any
# from IPython.display import clear_output

class ThreeWheeledRobotSystem:
    """System class: inverted pendulum. State transition function"""

    dim_action: int = 2
    dim_observation: int = 3
    dim_state: int = 3

    def __init__(self) -> None:
        """Initialize `InvertedPendulumSystem`"""

        self.reset()

    def reset(self) -> None:
        """Reset system to inital state."""

        self.action = np.zeros(self.dim_action)

    def compute_dynamics(self, state: np.array, action: np.array) -> np.array:
        """Calculate right-hand-side for Euler integrator

        Args:
            state (np.array): current state
            action (np.array): current action

        Returns:
            np.array: right-hand-side for Euler integrator
        """

        Dstate = np.zeros(self.dim_state)

        # -----------------------------------------------------------------------
        # HINT
        # Assume that Dstate is the right-hand side of the system dynamics
        # description, and assign proper values to the components of Dstate,
        # assuming that:
        #
        # Dstate[0] is \dot{x}_{rob}
        # Dstate[1] is \dot{y}_{rob}
        # Dstate[2] is \dot{\vartheta}

        # YOUR CODE GOES HERE

        Dstate[0] = action[0]*np.cos(state[2])
        Dstate[1] = action[0]*np.sin(state[2])
        Dstate[2] = action[1]

        # -----------------------------------------------------------------------

        return Dstate

    def compute_closed_loop_rhs(self, state: np.array) -> np.array:
        """Get right-hand-side for current observation and saved `self.action`

        Args:
            state (np.array): current state

        Returns:
            np.array: right-hand-side for Euler integrator
        """

        system_right_hand_side = self.compute_dynamics(state, self.action)
        return system_right_hand_side

    def receive_action(self, action: np.array) -> None:
        """Save current action to `self.action`

        Args:
            action (np.array): current action
        """

        self.action = action

    @staticmethod
    def get_observation(state: np.array) -> np.array:
        """Get observation given a state

        Args:
            state (np.array): system state

        Returns:
            np.array: observation
        """
        observation = state

        return observation

class Simulator:
    """Euler integrator"""

    def __init__(
        self,
        system: ThreeWheeledRobotSystem,
        N_steps: int,
        step_size: float,
        state_init: np.array,
    ):
        self.system = system
        self.N_steps = N_steps
        self.step_size = step_size
        self.state = np.copy(state_init)
        self.state_init = np.copy(state_init)
        self.current_step_idx = 0

    def step(self) -> bool:
        """Do one Euler integration step

        Returns:
            bool: status of simulation. `True` - simulation continues, `False` - simulation stopped
        """

        if self.current_step_idx <= self.N_steps:
            self.state += (
                self.system.compute_closed_loop_rhs(self.state) * self.step_size
            )
            self.current_step_idx += 1
            return True
        else:
            return False

    def reset(self) -> None:
        """Resets the system to initial state"""

        self.state = np.copy(self.state_init)
        self.current_step_idx = 0
        self.system.reset()

    def get_sim_step_data(self) -> Tuple[np.array, np.array, int]:
        """Get current observation, action and step id

        Returns:
            Tuple[np.array, np.array, int]:
        """

        return (
            self.system.get_observation(self.state),
            np.copy(self.system.action),
            int(self.current_step_idx),
        )

