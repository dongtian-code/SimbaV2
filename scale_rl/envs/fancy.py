"""fancy_gym BoxPushing support.

A 7-DoF Franka pushes a box to a target pose with normalised joint torques.
Observations are 29-D, actions 7-D, and an episode is a fixed 100 steps. The
FIRST observation element is the step counter, so the horizon is part of the
state and the fixed cut-off stays Markovian.

`BoxPushingEnvBase.step` ends the episode itself once `_steps` reaches 100,
splitting it into `terminated` when the box is within 5 cm / 0.5 rad of the
target and `truncated` otherwise. `configs/env/box_pushing.yaml` therefore
repeats `max_episode_steps: 100`: the repo's own TimeLimit wrapper cuts at the
same step (a no-op), and the gamma heuristic in `configs/online_rl.yaml` is
derived from it.

Why this does NOT `import fancy_gym`
------------------------------------
`fancy_gym/__init__.py` reaches `fancy_gym.utils.env_compatibility`, which
subclasses `gymnasium.wrappers.EnvCompatibility` -- removed in gymnasium 1.0 --
so on the gymnasium 1.1 / mujoco 3.3 stack this repo needs, `import fancy_gym`
dies before registering a single env id. That is the same incompatibility that
made `scale_rl/envs/metaworld.py` drop fancy_gym. The package is still perfectly
installable as long as pip is told not to resolve its dependencies (it pins
`mujoco==2.3.3`, which would drag the whole env back to MuJoCo 2):

    pip install --no-deps "git+https://github.com/DongTian95/fancy_gymnasium.git@dt_branch"

`_load_box_pushing_module` then loads the one env module straight off disk under
its canonical dotted name, with stub parent packages in `sys.modules` so its
intra-package import resolves and no `__init__.py` ever runs. The 18 MB of
Franka meshes stay in the installed package instead of being vendored here.

Kept in step with the sibling RLAC repo (`envs/box_pushing_utils.py`), which
does the same thing so the two projects can share one conda environment and run
the same task.
"""

import hashlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import types

import gymnasium as gym

# fancy_gym's own episode length, repeated in configs/env/box_pushing.yaml.
BOX_PUSHING_MAX_EPISODE_STEPS = 100

# Mirrors fancy_gym/envs/__init__.py: the "RandomInit" ids are the same reward
# classes built with `random_init=True`, which redraws the box's starting pose
# every reset instead of always starting it at (0.4, 0.3). The goal pose is
# resampled every reset either way.
BOX_PUSHING_ENVS = {
    "fancy/BoxPushingDense-v0": ("BoxPushingDense", {}),
    "fancy/BoxPushingTemporalSparse-v0": ("BoxPushingTemporalSparse", {}),
    "fancy/BoxPushingTemporalSpatialSparse-v0": ("BoxPushingTemporalSpatialSparse", {}),
    "fancy/BoxPushingRandomInitDense-v0": ("BoxPushingDense", {"random_init": True}),
    "fancy/BoxPushingRandomInitTemporalSparse-v0": (
        "BoxPushingTemporalSparse", {"random_init": True}),
    "fancy/BoxPushingRandomInitTemporalSpatialSparse-v0": (
        "BoxPushingTemporalSpatialSparse", {"random_init": True}),
}

# Parent packages of box_pushing_env, outermost first. Each gets a stub module
# with a `__path__` so the real `__init__.py` files never execute.
_STUB_PACKAGES = (
    ("fancy_gym", ()),
    ("fancy_gym.envs", ("envs",)),
    ("fancy_gym.envs.mujoco", ("envs", "mujoco")),
    ("fancy_gym.envs.mujoco.box_pushing", ("envs", "mujoco", "box_pushing")),
)

_MODULE_NAME = "fancy_gym.envs.mujoco.box_pushing.box_pushing_env"

# fancy_gym's model XMLs were written for MuJoCo 2.x. MuJoCo 3 dropped
# <option collision="...">, and its compiler rejects unknown attributes outright:
#     ValueError: XML Error: Schema violation: unrecognized attribute: 'collision'
# "all" was 2.x's default (check every geom pair), so deleting the attribute
# changes nothing about the simulation -- it is the only thing standing between
# this env and MuJoCo 3.3. The rewrite goes to a cache directory rather than into
# the installed package: fancy_gym is a shared checkout that the sibling dt_rl
# repo runs against on mujoco 2.3.3, and this keeps working against an
# unmodified upstream clone.
_XML_FIXUPS = (
    (re.compile(r'\s+collision="[^"]*"'), ""),
)


def is_fancy_env(env_name: str) -> bool:
    return env_name.startswith("fancy/")


def _patch_xml(text: str) -> str:
    for pattern, replacement in _XML_FIXUPS:
        text = pattern.sub(replacement, text)
    return text


def _asset_cache_root() -> str:
    override = os.environ.get("FANCY_GYM_ASSET_CACHE_DIR")
    if override:
        return override
    return os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "fancy_gym_mujoco3_assets",
    )


def _mujoco3_assets(pkg_dir: str) -> str:
    """Return an assets directory whose XMLs MuJoCo 3 will load.

    The package's own directory comes back unchanged when nothing needs
    patching, so a fancy_gym that has since been fixed upstream costs nothing.
    Otherwise the (35-file) tree is copied once into a content-addressed cache.
    Several envs racing to build it is fine -- the copy is staged and renamed
    into place, and the loser just reuses the winner's.
    """
    src = os.path.join(pkg_dir, "assets")
    patched = {}
    for root, _, files in os.walk(src):
        for name in sorted(files):
            if not name.endswith(".xml"):
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                text = f.read()
            new_text = _patch_xml(text)
            if new_text != text:
                patched[os.path.relpath(path, src)] = new_text
    if not patched:
        return src

    digest = hashlib.sha1(
        "\0".join(f"{k}\0{v}" for k, v in sorted(patched.items())).encode("utf-8")
    ).hexdigest()[:12]
    cache_root = _asset_cache_root()
    dst = os.path.join(cache_root, f"box_pushing_{digest}")
    if os.path.isdir(dst):
        return dst

    os.makedirs(cache_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"box_pushing_{digest}.", dir=cache_root)
    try:
        staged = os.path.join(tmp, "assets")
        shutil.copytree(src, staged)
        for rel, text in patched.items():
            with open(os.path.join(staged, rel), "w") as f:
                f.write(text)
        try:
            os.rename(staged, dst)
        except OSError:
            if not os.path.isdir(dst):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[box_pushing] MuJoCo-3 asset cache: {dst}", flush=True)
    return dst


def _exec_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `box_pushing_env` imports `box_pushing_utils`
    # by its dotted name, and the import machinery looks in sys.modules first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_box_pushing_module():
    """Return fancy_gym's `box_pushing_env` module, imported in isolation."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    # find_spec locates the package without executing its __init__.py.
    spec = importlib.util.find_spec("fancy_gym")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "fancy_gym is not installed, so the BoxPushing envs are unavailable. "
            "Install it WITHOUT its dependencies (it pins mujoco==2.3.3, which "
            "would downgrade this environment):\n"
            '    pip install --no-deps "git+https://github.com/DongTian95/fancy_gymnasium.git@dt_branch"'
        )
    root = list(spec.submodule_search_locations)[0]
    pkg_dir = os.path.join(root, "envs", "mujoco", "box_pushing")
    if not os.path.isdir(pkg_dir):
        raise ImportError(
            f"Found fancy_gym at {root!r}, but no box_pushing package at {pkg_dir!r}."
        )

    added = []
    try:
        for name, parts in _STUB_PACKAGES:
            if name in sys.modules:
                continue
            module = types.ModuleType(name)
            module.__path__ = [os.path.join(root, *parts)]
            module.__package__ = name
            sys.modules[name] = module
            added.append(name)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], leaf, module)
        # box_pushing_utils first: box_pushing_env imports names from it.
        for leaf in ("box_pushing_utils", "box_pushing_env"):
            name = f"fancy_gym.envs.mujoco.box_pushing.{leaf}"
            if name not in sys.modules:
                added.append(name)
                _exec_module(name, os.path.join(pkg_dir, f"{leaf}.py"))
    except Exception:
        # Leave sys.modules as it was found, so a later genuine `import
        # fancy_gym` is not silently served a half-built stub.
        for name in reversed(added):
            sys.modules.pop(name, None)
        raise

    return sys.modules[_MODULE_NAME]


class _Mujoco3Assets:
    """Load the model from `_MUJOCO3_XML` instead of the package's own copy.

    `BoxPushingEnvBase.__init__` hardcodes its `model_path`, so the redirect has
    to happen after `MujocoEnv.__init__` has resolved `self.fullpath` and before
    it compiles the model -- which is exactly `_initialize_simulation`.
    """

    _MUJOCO3_XML = None

    def _initialize_simulation(self):
        if self._MUJOCO3_XML is not None:
            self.fullpath = self._MUJOCO3_XML
        return super()._initialize_simulation()


class SuccessInfo(gym.Wrapper):
    """Expose BoxPushing's `is_success` flag under the `success` key as well.

    MetaWorld reports `success`, BoxPushing reports `is_success`, and
    `scale_rl/evaluation.py` only looks for `success` -- without this, a
    BoxPushing run would report `avg_success` of 0 forever. Both keys are kept,
    so the metric lines up with the MetaWorld runs and with RLAC's.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "is_success" in info:
            info["success"] = float(info["is_success"])
        return observation, reward, terminated, truncated, info


_ENV_CLASSES = {}


def _env_class(class_name: str):
    """fancy_gym's env class, subclassed to read the MuJoCo-3-patched model."""
    if class_name in _ENV_CLASSES:
        return _ENV_CLASSES[class_name]
    module = _load_box_pushing_module()
    base = getattr(module, class_name)
    pkg_dir = os.path.dirname(module.__file__)
    assets = _mujoco3_assets(pkg_dir)
    if assets == os.path.join(pkg_dir, "assets"):
        cls = base  # nothing needed patching
    else:
        cls = type(
            f"Mujoco3{class_name}",
            (_Mujoco3Assets, base),
            {"_MUJOCO3_XML": os.path.join(assets, "box_pushing.xml")},
        )
    _ENV_CLASSES[class_name] = cls
    return cls


def make_fancy_env(env_name: str, seed: int, **kwargs) -> gym.Env:
    """Build one BoxPushing env.

    `create_vec_env` adds RescaleAction (a no-op -- the action space is already
    [-1, 1]), TimeLimit and the action-repeat wrapper on top, and seeds the
    spaces. The env's own `np_random` -- which draws the box and goal poses -- is
    only seeded by a seeded reset, and the training loop resets without a seed,
    so it has to happen here.
    """
    if env_name not in BOX_PUSHING_ENVS:
        raise ValueError(
            f"Unknown fancy_gym env {env_name!r}. Expected one of: {sorted(BOX_PUSHING_ENVS)}."
        )
    class_name, env_kwargs = BOX_PUSHING_ENVS[env_name]
    env = _env_class(class_name)(**env_kwargs, **kwargs)
    env = SuccessInfo(env)
    env.reset(seed=seed)
    return env
