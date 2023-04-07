import pytest
from acme import environment_loop
from acme import specs
import dm_env
import numpy as np
from acme.testing import fakes
from torch import nn
from torchme import actors
import torch


class FakeNetwork(nn.Module):
    def __init__(self, env_spec: specs.EnvironmentSpec):
        super().__init__()
        obs_shape = env_spec.observations.shape
        flattened_obs_shape = obs_shape[0] * obs_shape[1]
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(flattened_obs_shape, env_spec.actions.num_values)
        )

    def forward(self, x):
        out = self.network(x)
        out = torch.argmax(out)
        return out.to(torch.int32)


@pytest.fixture
def fake_env() -> dm_env.Environment:
    env_spec = specs.EnvironmentSpec(
        observations=specs.Array(shape=(10, 5), dtype=np.float32),
        actions=specs.DiscreteArray(num_values=3),
        rewards=specs.Array(shape=(), dtype=np.float32),
        discounts=specs.BoundedArray(
            shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
        ),
    )
    return fakes.Environment(env_spec, episode_length=10)


def test_feedforward(fake_env):
    env_spec = specs.make_environment_spec(fake_env)

    network = FakeNetwork(env_spec)
    actor = actors.FeedForwardActor(network)
    loop = environment_loop.EnvironmentLoop(fake_env, actor)
    loop.run(20)
