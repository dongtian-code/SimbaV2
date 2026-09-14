from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

import wandb


def evaluate(
    agent,
    env: VectorEnv,
    num_episodes: int,
) -> Dict[str, float]:
    n = env.num_envs

    assert num_episodes % n == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // n

    total_returns = []
    total_successes = []
    total_lengths = []

    for _ in range(num_eval_episodes_per_env):
        returns = np.zeros(n)
        lengths = np.zeros(n)
        successes = np.zeros(n)

        observations, infos = env.reset()

        prev_timestep = {"next_observation": observations}

        dones = np.zeros(n)
        while np.sum(dones) < n:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_timestep=prev_timestep,
                training=False,
            )
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions
            )

            prev_timestep = {"next_observation": next_observations}

            returns += rewards * (1 - dones)
            lengths += 1 - dones

            # Success is taken from the LAST step of each episode -- the value at
            # the step where that sub-env terminated or truncated -- NOT "did it
            # succeed at any point". This is the stricter reading: on tasks where
            # the object can be pushed into the goal region and then drift back
            # out (push, sweep, soccer, plate-slide) a run only counts if it is
            # still solved when the episode ends. It also matches the sibling
            # RLAC repo, whose `evaluation.py` reads the final info once after
            # its rollout loop -- the two numbers have to mean the same thing to
            # be plotted against each other.
            #
            # Where the ending step's info lives depends on autoreset. Under
            # gymnasium's SameStep autoreset (see `create_vec_env`) the vector
            # env resets in the same step it ends, so the top-level `infos`
            # describes the *already reset* env and the real final info is moved
            # into `infos["final_info"]` (a dict of batched arrays, masked by
            # `infos["_final_info"]`). final_info therefore has to be checked
            # first; the plain-`infos` branch covers ordinary mid-episode steps
            # and any autoreset mode that reports the final info in place.
            step_success = None
            final_info = infos.get("final_info")
            if isinstance(final_info, dict) and "success" in final_info:
                step_success = np.asarray(final_info["success"], dtype=float).reshape(n)
                ended = np.asarray(
                    infos.get("_final_info", np.ones(n, dtype=bool))
                ).reshape(n).astype(bool)
            elif "success" in infos:
                step_success = np.asarray(infos["success"], dtype=float).reshape(n)
                ended = np.ones(n, dtype=bool)

            if step_success is not None:
                # Overwrite while the sub-env is still running and freeze it once
                # done, so what survives is that sub-env's final-step value.
                successes = np.where(ended & (dones == 0), step_success, successes)

            # once an episode is done in a sub-environment, we assume it to be done.
            # also, we assume to be done whether it is terminated or truncated during evaluation.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            observations = next_observations

        for env_idx in range(n):
            total_returns.append(returns[env_idx])
            total_lengths.append(lengths[env_idx])
            # already the final step's 0/1 value -- no "ever succeeded" collapse
            total_successes.append(float(successes[env_idx]))

    eval_info = {
        "avg_return": np.mean(total_returns),
        "avg_length": np.mean(total_lengths),
        "avg_success": np.mean(total_successes),
    }

    return eval_info


def record_video(
    agent,
    env: VectorEnv,
    num_episodes: int,
    video_length: int = 100,
) -> Dict[str, float]:
    n = env.num_envs
    assert num_episodes % n == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // n

    total_videos = []

    for _ in range(num_eval_episodes_per_env):
        videos = []

        observations, infos = env.reset()
        prev_timestep = {"next_observation": observations}
        images = env.call("render")
        dones = np.zeros(n)
        while np.sum(dones) < n:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_timestep=prev_timestep,
                training=False,
            )
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions
            )

            prev_timestep = {"next_observation": next_observations}

            # once an episode is done in a sub-environment, we assume it to be done.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            videos.append(images)
            images = env.call("render")
            observations = next_observations

        total_videos.append(np.stack(videos, axis=1))  # (n, t, c, h, w)

    total_videos = np.concatenate(total_videos, axis=0)  # (b, t, h, w, c)
    total_videos = total_videos[:, :video_length]
    total_videos = total_videos.transpose(0, 1, 4, 2, 3)  # (b, t, c, h, w)

    video_info = {"video": wandb.Video(total_videos, fps=10, format="gif")}

    return video_info
