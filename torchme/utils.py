import torch
import tree
from acme.types import Transition


def transition_tensors_to_torch(transitions: Transition, device: str):
    transitions = tree.map_structure(
        lambda x: torch.as_tensor(x.numpy(), dtype=torch.float32, device=device),
        transitions,
    )
    # observation = transitions.observation
    # Unsqueeze if obs is scalar
    if len(transitions.observation.shape) == 1:
        transitions = transitions._replace(
            observation=transitions.observation.unsqueeze(-1),
            next_observation=transitions.next_observation.unsqueeze(-1),
        )
    return transitions
