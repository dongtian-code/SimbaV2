"""Full-state checkpointing for preemptible (SLURM requeue) training runs.

This is deliberately separate from `BaseAgent.save()/load()`. Those write an
orbax checkpoint per network and are meant for *transferring weights* -- the
default `load_only_param: true` drops the optimizer state entirely, which is
right for fine-tuning and wrong for resuming. A job that gets preempted at step
3.2M has to come back bit-for-bit: parameters, Adam moments, the agent's PRNG
key, the observation/reward normalizer statistics and the replay buffer.

Everything is written as one pickle so a checkpoint is a single atomic file --
`os.replace` onto the final name means a job killed mid-write leaves the
previous checkpoint intact rather than a half-written directory tree.

Layout of the pickle:
    {
        'agent': {'networks': {...}, 'rng': ndarray, 'wrappers': {...}},
        'buffer': <buffer.state_dict()> or None,
        'extra':  <caller-supplied bookkeeping: step counters, wandb run id, ...>,
    }

A small `<checkpoint>.meta` sidecar holding just `extra` is written alongside it.
Checkpoint discovery has to inspect every candidate file before it knows which
one to resume from, and a full checkpoint is dominated by the replay buffer
(hundreds of MB); reading the sidecar keeps that scan to a few hundred bytes.
The sidecar is written *after* the checkpoint, so its presence also certifies
that the checkpoint next to it is complete.
"""

import os
import pickle

import flax
import jax.numpy as jnp
import numpy as np

from scale_rl.agents.base_agent import AgentWrapper
from scale_rl.agents.wrappers import ObservationNormalizer, RewardNormalizer

# The `flax.struct.dataclass` Network attributes a SimbaV2/Simba agent owns.
# `flax.serialization.to_state_dict` on a Network captures exactly its pytree
# fields -- params, opt_state and update_step -- while `network_def` and `tx`
# stay static and are rebuilt by `create_agent`.
_NETWORK_ATTRS = ("_actor", "_critic", "_target_critic", "_temperature")


def agent_state_dict(agent) -> dict:
    networks = {}
    wrappers = {}

    node = agent
    while isinstance(node, AgentWrapper):
        if isinstance(node, ObservationNormalizer):
            wrappers["observation"] = {
                "mean": np.array(node.obs_rms.mean),
                "var": np.array(node.obs_rms.var),
                "count": node.obs_rms.count,
            }
        elif isinstance(node, RewardNormalizer):
            wrappers["reward"] = {
                "G": np.array(node.G),
                "mean": np.array(node.G_rms.mean),
                "var": np.array(node.G_rms.var),
                "count": node.G_rms.count,
                "G_r_max": float(node.G_r_max),
            }
        node = node.agent

    for attr in _NETWORK_ATTRS:
        network = getattr(node, attr, None)
        if network is None:
            continue
        networks[attr] = flax.serialization.to_state_dict(network)

    rng = getattr(node, "_rng", None)
    return {
        "networks": networks,
        "rng": None if rng is None else np.asarray(rng),
        "wrappers": wrappers,
    }


def load_agent_state_dict(agent, state: dict) -> None:
    """Restore in place into a freshly built agent of the same configuration."""
    wrappers = state.get("wrappers", {})

    node = agent
    while isinstance(node, AgentWrapper):
        if isinstance(node, ObservationNormalizer) and "observation" in wrappers:
            saved = wrappers["observation"]
            node.obs_rms.mean = np.array(saved["mean"])
            node.obs_rms.var = np.array(saved["var"])
            node.obs_rms.count = saved["count"]
        elif isinstance(node, RewardNormalizer) and "reward" in wrappers:
            saved = wrappers["reward"]
            node.G = np.array(saved["G"])
            node.G_rms.mean = np.array(saved["mean"])
            node.G_rms.var = np.array(saved["var"])
            node.G_rms.count = saved["count"]
            node.G_r_max = float(saved["G_r_max"])
        node = node.agent

    for attr, network_state in state.get("networks", {}).items():
        network = getattr(node, attr, None)
        if network is None:
            raise ValueError(
                f"Checkpoint holds a {attr!r} network but the rebuilt agent has none; "
                "the agent configuration changed since the checkpoint was written."
            )
        # from_state_dict uses `network` as the target, so the static fields
        # (network_def, tx) come from the live agent and only the pytree leaves
        # are replaced -- shapes must match, and a mismatch raises here.
        setattr(node, attr, flax.serialization.from_state_dict(network, network_state))

    if state.get("rng") is not None:
        node._rng = jnp.asarray(state["rng"])


META_SUFFIX = ".meta"


def _atomic_pickle(path: str, payload) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_checkpoint(path: str, agent, buffer=None, extra: dict = None) -> None:
    """Atomically write agent + replay buffer + caller bookkeeping to `path`."""
    extra = extra or {}
    _atomic_pickle(
        path,
        {
            "agent": agent_state_dict(agent),
            "buffer": (
                buffer.state_dict()
                if buffer is not None and hasattr(buffer, "state_dict")
                else None
            ),
            "extra": extra,
        },
    )
    # Written second on purpose: a reader that finds the sidecar knows the
    # checkpoint beside it finished writing.
    _atomic_pickle(path + META_SUFFIX, extra)


def load_checkpoint(path: str, agent, buffer=None) -> dict:
    """Restore a checkpoint written by `save_checkpoint` into `agent`/`buffer`.

    Returns the `extra` bookkeeping dict. The agent is mutated in place (its
    networks are immutable flax structs, but the agent object holding them is
    not), so there is no agent to return.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    load_agent_state_dict(agent, payload["agent"])

    if buffer is not None and payload.get("buffer") is not None:
        if not hasattr(buffer, "load_state_dict"):
            raise TypeError(f"{type(buffer).__name__} does not support load_state_dict")
        buffer.load_state_dict(payload["buffer"])

    print(f"[checkpoint] Restored checkpoint from {path}", flush=True)
    return payload.get("extra", {})


def read_checkpoint_extra(path: str):
    """Read only the bookkeeping dict, or None if the checkpoint is unusable.

    Used by checkpoint discovery, which has to inspect many candidate files
    before an agent exists to restore into. Prefers the `.meta` sidecar and only
    falls back to unpickling the full checkpoint (replay buffer included) for
    checkpoints written before the sidecar existed.
    """
    meta_path = path + META_SUFFIX
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "rb") as f:
                extra = pickle.load(f)
            if isinstance(extra, dict):
                return extra
        except Exception as error:
            print(
                f"[checkpoint] Ignoring unreadable checkpoint sidecar {meta_path}: {error}",
                flush=True,
            )

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as error:  # truncated / half-written / stale format
        print(f"[checkpoint] Ignoring unreadable checkpoint {path}: {error}", flush=True)
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("extra", {}) or {}
