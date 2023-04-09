import torch
from acme import core


class FeedForwardActor(core.Actor):
    def __init__(self, policy_network, env_spec, adder=None, variable_client=None):
        self.policy_network = policy_network
        self.env_spec = env_spec
        self.variable_client = variable_client
        self.adder = adder
        self.device = next(self.policy_network.parameters()).device

    def select_action(self, observation):
        self.policy_network.eval()

        observation = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        policy = self.policy_network(observation)
        action = (
            policy.sample()
            if isinstance(policy, torch.distributions.Distribution)
            else policy
        )
        return action.cpu().detach().numpy().astype(self.env_spec.actions.dtype)

    def observe_first(self, timestep):
        if self.adder:
            self.adder.add_first(timestep)

    def observe(self, action, next_timestep):
        if self.adder:
            self.adder.add(action, next_timestep)

    def update(self, wait=False):
        if self.variable_client:
            self.variable_client.update(wait)
