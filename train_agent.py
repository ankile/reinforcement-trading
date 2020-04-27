import os
from datetime import datetime

import numpy as np
import pandas as pd

import configuration as conf
import gym
import ptan
import torch
import torch.optim as optim
from gym import wrappers
from lib import common, data, environ, models, validation


def train_agent(
    run_name,
    data_paths=conf.default_data_paths,
    validation_paths=conf.default_validation_paths,
    model=models.DQNConv1D,
    large=False,
    load_checkpoint=None,
    saves_path=None,
    eps_steps=None,
):
    """
    Main function for training the agents

    :run_name: a string of choice that dictates where to save
    :data_paths: dict specifying what data to train with
    :validation_paths: dict specifying what data to validate with
    :model: what model to use
    :large: whether or not to use large feature set
    :load_checkpoint: an optinal path to checkpoint to load from
    """

    print("=" * 80)
    print("Training starting".rjust(40 + 17 // 2))
    print("=" * 80)

    # Get training data
    stock_data = data.get_data_as_dict(data_paths, large=large)
    val_data = data.get_data_as_dict(validation_paths, large=large)

    # Setup before training can begin
    step_idx = 0
    eval_states = None
    best_mean_val = None
    EPSILON_STEPS = eps_steps if eps_steps is not None else conf.EPSILON_STEPS

    # Use GPU if available, else fall back on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Set up the path to save the checkpoints to
    if saves_path is None:
        saves_path = os.path.join("saves", run_name)
    else:
        saves_path = os.path.join(saves_path, run_name)

    print(f"[Info] Saving to path: {saves_path}")

    os.makedirs(saves_path, exist_ok=True)

    # Create the gym-environment that the agent will interact with during training
    env = environ.StocksEnv(
        stock_data,
        bars_count=conf.BARS_COUNT,
        reset_on_close=conf.RESET_ON_CLOSE,
        random_ofs_on_reset=conf.RANDOM_OFS_ON_RESET,
        reward_on_close=conf.REWARD_ON_CLOSE,
        large=large,
    )

    env = wrappers.TimeLimit(env, max_episode_steps=1000)

    # Create the gym-environment that the agent will interact with when validating
    env_val = environ.StocksEnv(
        val_data,
        bars_count=conf.BARS_COUNT,
        reset_on_close=conf.RESET_ON_CLOSE,
        random_ofs_on_reset=conf.RANDOM_OFS_ON_RESET,
        reward_on_close=conf.REWARD_ON_CLOSE,
        large=large,
    )

    # Create the model
    net = model(env.observation_space.shape, env.action_space.n).to(device)

    print("Using network:".rjust(40 + 14 // 2))
    print("=" * 80)
    print(net)

    # Initialize agent and epsilon-greedy action-selector from the ptan package
    # The ptan package provides some helper and wrapper functions for ease of
    # use of reinforcement learning
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(conf.EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, conf.GAMMA, steps_count=conf.REWARD_STEPS
    )
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, conf.REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=conf.LEARNING_RATE)

    # If a checkpoint is supplied to the function –> resume the training from there
    if load_checkpoint is not None:
        state = torch.load(load_state)
        net.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step_idx = state["step_idx"]
        best_mean_val = state["best_mean_val"]
        print(f"State loaded –> step index: {step_idx}, best mean val: {best_mean_val}")

        net.train()

    # Create a reward tracker, i.e. an object that keeps track of the
    # rewards the agent gets during training
    reward_tracker = common.RewardTracker(np.inf, group_rewards=100)

    # The main training loop
    print("Training loop starting".rjust(40 + 22 // 2))
    print("=" * 80)

    # Run the main training loop
    while True:
        step_idx += 1
        buffer.populate(1)

        # Get current epsilon for epsilon-greedy action-selection
        selector.epsilon = max(
            conf.EPSILON_STOP, conf.EPSILON_START - step_idx / EPSILON_STEPS
        )

        # Take a step and get rewards
        new_rewards = exp_source.pop_rewards_steps()
        if new_rewards:
            reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

        # As long as not enough data is in buffer, go to top again
        if len(buffer) < conf.REPLAY_INITIAL:
            continue

        if eval_states is None:
            print("Initial buffer populated, start training")
            eval_states = buffer.sample(conf.STATES_TO_EVALUATE)
            eval_states = [
                np.array(transition.state, copy=False) for transition in eval_states
            ]
            eval_states = np.array(eval_states, copy=False)

        # Evaluate the model every x number of steps
        # and update the currently best performance if better value gotten
        if step_idx % conf.EVAL_EVERY_STEP == 0:
            mean_val = common.calc_values_of_states(eval_states, net, device=device)
            # If new best value –> save the model, both with meta data for resuming training
            # and as the full object for use in testing
            if best_mean_val is None or best_mean_val < mean_val:
                if best_mean_val is not None:
                    print(
                        f"{step_idx}: Best mean value updated {best_mean_val:.3f} -> {mean_val:.3f}"
                    )
                best_mean_val = mean_val
                # Save checkpoint with meta data
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step_idx": step_idx,
                        "best_mean_val": best_mean_val,
                    },
                    os.path.join(saves_path, f"mean_val-{mean_val:.3f}.data"),
                )
                # Save full object for testing
                torch.save(
                    net,
                    os.path.join(saves_path, f"mean_val-{mean_val:.3f}-fullmodel.data"),
                )

        # Reset optimizer's gradients before optimization step
        optimizer.zero_grad()
        batch = buffer.sample(conf.BATCH_SIZE)
        # Calculate the loss
        loss_v = common.calc_loss(
            batch,
            net,
            tgt_net.target_model,
            conf.GAMMA ** conf.REWARD_STEPS,
            device=device,
        )
        # Calculate the gradient
        loss_v.backward()
        # Do one step of gradient descent
        optimizer.step()

        # Sync up the to networks we're using
        # Two networks in this manner should increase the agent's ability to converge
        if step_idx % conf.TARGET_NET_SYNC == 0:
            tgt_net.sync()

        # Every 1 million steps, save model in case something happens
        # so we can resume training in that case
        if step_idx % conf.CHECKPOINT_EVERY_STEP == 0:
            idx = step_idx // conf.CHECKPOINT_EVERY_STEP
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step_idx": step_idx,
                    "best_mean_val": best_mean_val,
                },
                os.path.join(saves_path, f"checkpoint-{idx}.data"),
            )
            torch.save(net, os.path.join(saves_path, f"fullmodel-{idx}.data"))

    print("Training done")


# This is the entry point for the code if this file is run directly
if __name__ == "__main__":
    # Run the training code
    train_agent(
        "test-local-2",
        data_paths=conf.default_data_paths,
        validation_paths=conf.default_validation_paths,
        model=models.DQNConv1D,
        large=False,
        load_checkpoint=None,
    )
