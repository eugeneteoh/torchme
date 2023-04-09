import copy

import acme.adders.reverb as reverb_adders
import reverb
from acme import datasets
from acme.agents import agent

from torchme import actors
from torchme.dqn import learning
from torchme.networks.policy import EpsilonGreedy


class DQN(agent.Agent):
    def __init__(
        self,
        environment_spec,
        network,
        batch_size,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        importance_sampling_exponent: float = 0.2,
        priority_exponent: float = 0.6,
        n_step: int = 5,
        epsilon=0.05,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        logger=None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/acme",
        policy_network=None,
        max_gradient_norm=1e10,
        device="cpu",
    ):
        network.to(device)
        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = reverb.Table(
            name=reverb_adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=reverb_adders.NStepTransitionAdder.signature(environment_spec),
        )
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f"localhost:{self._server.port}"
        adder = reverb_adders.NStepTransitionAdder(
            client=reverb.Client(address), n_step=n_step, discount=discount
        )

        # The dataset provides an interface to sample from replay.
        replay_client = reverb.Client(address)
        dataset = datasets.make_reverb_dataset(
            server_address=address, batch_size=batch_size, prefetch_size=prefetch_size
        )

        # Create epsilon greedy policy network by default.
        if policy_network is None:
            # Use constant 0.05 epsilon greedy policy by default.
            policy_network = EpsilonGreedy(network, epsilon=epsilon)

        # Create a target network.
        target_network = copy.deepcopy(network)

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(policy_network, environment_spec, adder)

        # The learner updates the parameters (and initializes them).
        learner = learning.DQNLearner(
            network=network,
            target_network=target_network,
            discount=discount,
            importance_sampling_exponent=importance_sampling_exponent,
            learning_rate=learning_rate,
            target_update_period=target_update_period,
            dataset=dataset,
            replay_client=replay_client,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            checkpoint=checkpoint,
            save_directory=checkpoint_subpath,
        )

        # TODO: Add checkpoint for torch models
        # if checkpoint:
        #     self._checkpointer = tf2_savers.Checkpointer(
        #         directory=checkpoint_subpath,
        #         objects_to_save=learner.state,
        #         subdirectory='dqn_learner',
        #         time_delta_minutes=60.)
        # else:
        #     self._checkpointer = None

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
        )

    def update(self):
        super().update()
        # if self._checkpointer is not None:
        #     self._checkpointer.save()
