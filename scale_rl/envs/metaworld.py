"""MetaWorld (ML1 / 50-task benchmark) support.

MetaWorld 3.x registers its own gymnasium entry points as a side effect of
`import metaworld`. "Meta-World/goal_observable" is the single-task,
goal-in-the-observation variant used here; the concrete task is passed through
an `env_name` kwarg. Observations are 39-D, actions 4-D, and every task runs a
fixed 500-step episode -- MetaWorld never terminates early, it only truncates
(which is why `configs/env/metaworld.yaml` sets `episodic: false`).

This deliberately does *not* go through `fancy_gym`: fancy_gym targets
gymnasium 0.29 (it subclasses the `EnvCompatibility` wrapper gymnasium 1.0
removed) and pins mujoco==2.3.3, neither of which is compatible with the
gymnasium 1.1 / mujoco 3.3 stack this repo needs. The same decision was made in
the RLAC repo, so both projects can share one conda environment.

Task ids accept every spelling the sibling RLAC configs use -- "assembly",
"assembly-v2", "assembly-v3" and "metaworld/assembly-v2" all resolve to
MetaWorld 3.x's "assembly-v3". Keeping the "metaworld/<task>-v2" spelling in
configs means a SimbaV2 run and an RLAC run of the same task carry the same
`env_name` in W&B, which is what makes them directly comparable in plots.
"""

import re

import gymnasium as gym

# The gymnasium id MetaWorld 3.x registers for a single task whose goal is part
# of the observation.
_GOAL_OBSERVABLE_ID = "Meta-World/goal_observable"

# MetaWorld's own episode length; `configs/env/metaworld.yaml` repeats it as
# `max_episode_steps` because the repo's TimeLimit wrapper needs it explicitly
# and the gamma heuristic in `configs/online_rl.yaml` is derived from it.
METAWORLD_MAX_EPISODE_STEPS = 500

# The 50 ML1 tasks, in the "metaworld/<task>-v2" spelling shared with RLAC.
METAWORLD_MT50 = [
    "metaworld/assembly-v2",
    "metaworld/basketball-v2",
    "metaworld/bin-picking-v2",
    "metaworld/box-close-v2",
    "metaworld/button-press-topdown-v2",
    "metaworld/button-press-topdown-wall-v2",
    "metaworld/button-press-v2",
    "metaworld/button-press-wall-v2",
    "metaworld/coffee-button-v2",
    "metaworld/coffee-pull-v2",
    "metaworld/coffee-push-v2",
    "metaworld/dial-turn-v2",
    "metaworld/disassemble-v2",
    "metaworld/door-close-v2",
    "metaworld/door-lock-v2",
    "metaworld/door-open-v2",
    "metaworld/door-unlock-v2",
    "metaworld/drawer-close-v2",
    "metaworld/drawer-open-v2",
    "metaworld/faucet-open-v2",
    "metaworld/faucet-close-v2",
    "metaworld/hammer-v2",
    "metaworld/hand-insert-v2",
    "metaworld/handle-press-side-v2",
    "metaworld/handle-press-v2",
    "metaworld/handle-pull-side-v2",
    "metaworld/handle-pull-v2",
    "metaworld/lever-pull-v2",
    "metaworld/peg-insert-side-v2",
    "metaworld/pick-place-wall-v2",
    "metaworld/pick-out-of-hole-v2",
    "metaworld/reach-v2",
    "metaworld/push-back-v2",
    "metaworld/push-v2",
    "metaworld/pick-place-v2",
    "metaworld/plate-slide-v2",
    "metaworld/plate-slide-side-v2",
    "metaworld/plate-slide-back-v2",
    "metaworld/plate-slide-back-side-v2",
    "metaworld/peg-unplug-side-v2",
    "metaworld/push-wall-v2",
    "metaworld/soccer-v2",
    "metaworld/stick-push-v2",
    "metaworld/stick-pull-v2",
    "metaworld/reach-wall-v2",
    "metaworld/shelf-place-v2",
    "metaworld/sweep-into-v2",
    "metaworld/sweep-v2",
    "metaworld/window-open-v2",
    "metaworld/window-close-v2",
]


def _metaworld():
    """Import MetaWorld, which registers its "Meta-World/..." gymnasium ids as an
    import side effect, and return the module.

    Deferred rather than imported at module scope so that `import scale_rl.envs`
    keeps working in an environment that has no MetaWorld installed.
    """
    import metaworld

    return metaworld


def metaworld_task_name(env_name: str) -> str:
    """Normalise any accepted spelling onto MetaWorld 3.x's "<task>-v3"."""
    metaworld = _metaworld()

    task = env_name.split("/", 1)[-1]
    task = re.sub(r"-v\d+$", "", task) + "-v3"
    if task not in metaworld.ALL_V3_ENVIRONMENTS:
        raise ValueError(
            f"Unknown MetaWorld task {task!r} (from env_name {env_name!r}). "
            f"Expected one of the {len(metaworld.ALL_V3_ENVIRONMENTS)} tasks in "
            "metaworld.ALL_V3_ENVIRONMENTS."
        )
    return task


def make_metaworld_env(
    env_name: str,
    seed: int,
    **kwargs,
) -> gym.Env:
    # gym.make below can only resolve "Meta-World/goal_observable" once importing
    # metaworld has registered it.
    _metaworld()
    task = metaworld_task_name(env_name)

    # disable_env_checker: MetaWorld declares loose observation-space bounds, so
    # gymnasium's passive checker warns on every reset/step ("obs ... is not
    # within the observation space") and pays for a space check per step. Both
    # are pure overhead here.
    env = gym.make(
        _GOAL_OBSERVABLE_ID,
        env_name=task,
        seed=seed,
        render_mode="rgb_array",
        disable_env_checker=True,
    )

    return env
