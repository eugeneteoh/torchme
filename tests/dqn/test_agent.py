# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tests for DQN agent."""

import acme
import numpy as np
from acme import specs
from acme.testing import fakes
from torch import nn

from torchme import dqn


class FakeNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 50), nn.ReLU(), nn.Linear(50, num_actions)
        )

    def forward(self, x):
        out = self.network(x)
        # out = torch.argmax(out)
        return out


def test_dqn():
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(3,),
        obs_dtype=np.float32,
        episode_length=10,
    )
    spec = specs.make_environment_spec(environment)
    # Construct the agent.
    agent = dqn.DQN(
        environment_spec=spec,
        network=FakeNetwork(spec.actions.num_values),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)
