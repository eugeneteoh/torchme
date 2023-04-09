import time
from typing import List, Optional, Union

import acme
import acme.adders.reverb as reverb_adders
import numpy as np
import reverb
import tensorflow as tf
import torch
import torch.nn.functional as F
import tree
from acme.utils import counting, loggers
from torch import nn, optim

from torchme.utils import transition_tensors_to_torch


class DQNLearner(acme.Learner):
    def __init__(
        self,
        network: nn.Module,
        target_network: nn.Module,
        discount: float,
        importance_sampling_exponent: float,
        learning_rate: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        max_abs_reward: Optional[float] = 1.0,
        huber_loss_parameter: float = 1.0,
        replay_client: Optional[Union[reverb.Client, reverb.TFClient]] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        save_directory: str = "~/acme",
        max_gradient_norm: Optional[float] = 1e10,
    ):
        self.network = network
        self.target_network = target_network
        self.discount = discount
        self.importance_sampling_exponent = importance_sampling_exponent
        self.target_update_period = target_update_period
        self.iterator = iter(dataset)
        self.max_abs_reward = max_abs_reward
        self.huber_loss_parameter = huber_loss_parameter
        self.replay_client = replay_client
        self.counter = counter or counting.Counter()
        self.logger = logger or loggers.TerminalLogger("learner", time_delta=1.0)
        self.max_gradient_norm = max_gradient_norm

        self.device = next(network.parameters()).device

        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self._variables = list(network.parameters())
        self._num_steps = 0
        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = 0

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        return tree.map_structure(lambda x: x.detach().cpu().numpy(), self._variables)

    def _step(self):
        self.network.train()

        inputs = next(self.iterator)
        transitions = inputs.data
        keys, probs = inputs.info[:2]
        keys, probs = keys.numpy(), torch.as_tensor(probs.numpy(), device=self.device)

        # Convert tf.Tensor to torch.Tensor
        transitions = transition_tensors_to_torch(transitions, self.device)

        q_tm1 = self.network(transitions.observation)
        q_t_value = self.target_network(transitions.next_observation)
        q_t_selector = self.network(transitions.next_observation)

        if self.max_abs_reward:
            r_t = torch.clamp(
                transitions.reward, -self.max_abs_reward, self.max_abs_reward
            )
        d_t = transitions.discount * self.discount

        # Double Q-learning
        best_action = torch.argmax(q_t_selector, dim=1)
        with torch.no_grad():
            double_q_bootstrapped = torch.gather(
                q_t_value, 1, best_action.unsqueeze(1)
            ).squeeze()
            target = r_t + d_t * double_q_bootstrapped

        qa_tm1 = torch.gather(
            q_tm1, 1, transitions.action.to(torch.int64).unsqueeze(1)
        ).squeeze()
        # qa_tm1 = qa_tm1.to(torch.float32)

        loss = F.huber_loss(
            qa_tm1, target, reduction="none", delta=self.huber_loss_parameter
        )

        # Importance weights
        importance_weights = 1 / probs
        importance_weights **= self.importance_sampling_exponent
        importance_weights /= torch.max(importance_weights)

        # Weighted-average Huber loss
        loss *= importance_weights
        loss = torch.mean(loss)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.max_gradient_norm
        )
        self.optimizer.step()

        # Priorities will be the absolute of the TD error
        priorities = torch.abs(target - qa_tm1)

        # Periodically update the target network
        # Hard update
        if self._num_steps % self.target_update_period == 0:
            for param, target_param in zip(
                self.network.parameters(), self.target_network.parameters()
            ):
                param.data.copy_(target_param.data)

        self._num_steps += 1

        # Report loss & statistics for logging
        fetches = {
            "loss": loss.cpu().detach().item(),
            "keys": keys,
            "priorities": priorities.cpu().detach().numpy(),
        }

        return fetches

    def step(self):
        result = self._step()

        keys = result.pop("keys")
        priorities = result.pop("priorities")

        # Update priorities in replay buffer
        if self.replay_client:
            self.replay_client.mutate_priorities(
                table=reverb_adders.DEFAULT_PRIORITY_TABLE,
                updates=dict(zip(keys, priorities)),
            )

        # Compute elapsed time
        timestamp = time.monotonic()
        elapsed_time = timestamp - self._timestamp
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self.counter.increment(steps=1, walltime=elapsed_time)
        result.update(counts)

        # Snapshot and attempt to write logs.
        # if self._snapshotter is not None:
        # self._snapshotter.save()
        self.logger.write(result)
