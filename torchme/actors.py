from acme import core
import torch


class FeedForwardActor(core.Actor):
    def __init__(self, policy_network, adder=None, variable_client=None):
        self.policy_network = policy_network
        self.variable_client = variable_client
        self.adder = adder

    def select_action(self, observation):
        observation = torch.as_tensor(observation).unsqueeze(0)
        policy = self.policy_network(observation)
        action = (
            policy.sample()
            if isinstance(policy, torch.distributions.Distribution)
            else policy
        )
        return action.cpu().detach().numpy()

    def observe_first(self, timestep):
        if self.adder:
            self.adder.add_first(timestep)

    def observe(self, action, next_timestep):
        if self.adder:
            self.adder.add(action, next_timestep)

    def update(self, wait=False):
        if self.variable_client:
            self.variable_client.update(wait)
