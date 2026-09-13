#!/usr/bin/env bash
set -Eeuo pipefail

# Builds the Conda environment the cw2 / MetaWorld runs use on DESY Maxwell.
# Everything lives inside the env (conda create, then pip installs), so nothing
# needs sudo or system-wide write access.
#
#   bash deps/setup_maxwell.sh                     # CUDA build, env "simbav2_env"
#   SIMBAV2_CUDA=0 bash deps/setup_maxwell.sh      # CPU-only JAX (laptop / login node)
#   ENV_NAME=my_env bash deps/setup_maxwell.sh     # different env name
#
# The env name must match the CONDA_PREFIX in configs/cw2/metaworld_online.yml.

ENV_NAME="${ENV_NAME:-simbav2_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SIMBAV2_CUDA="${SIMBAV2_CUDA:-1}"
SIMBAV2_DEPS_DIR="${SIMBAV2_DEPS_DIR:-}"

read -r -a CONDA_CHANNEL_ARGS <<< "${CONDA_CHANNEL_ARGS:---override-channels -c conda-forge}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_PARENT="$(cd -- "$REPO_ROOT/.." && pwd)"
DEPS_DIR="${SIMBAV2_DEPS_DIR:-$REPO_PARENT/.deps}"

log()  { printf '\n[simbav2 setup] %s\n' "$*"; }
die()  { printf '\n[simbav2 setup error] %s\n' "$*" >&2; exit 1; }
run()  { printf '+'; printf ' %q' "$@"; printf '\n'; "$@"; }

trap 'die "Setup failed near line $LINENO. Fix the error above and rerun."' ERR

init_conda() {
    command -v conda >/dev/null 2>&1 || die "conda not found on PATH."
    local base; base="$(conda info --base)" || die "Could not locate the Conda base installation."
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$base/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)"
    fi
}

# Reuse an existing shared checkout, or clone it if absent, then pip install -e it.
#
# Deliberately NON-DESTRUCTIVE: an existing checkout is installed at whatever
# commit it is already on, with no fetch/checkout/pull. $DEPS_DIR is shared with
# the sibling dt_rl and RLAC repos, their editable installs resolve to this
# original path (not to cw2's code copy), and their jobs re-import from it on
# every requeue. Moving this checkout would change the code under experiments
# that are currently running. Update it by hand, deliberately, when no jobs
# depend on it.
clone_or_reuse_repo() {
    local name="$1" url="$2" branch="$3" target="$DEPS_DIR/$name"
    mkdir -p "$DEPS_DIR"
    if [[ -d "$target/.git" ]]; then
        log "Reusing the existing $name checkout at $target (NOT updating it)."
        run git -C "$target" log --oneline -1
        run git -C "$target" status --short --branch | head -1
    else
        log "Cloning $name into $target at branch $branch."
        run git clone --branch "$branch" "$url" "$target"
    fi
    run python -m pip install -e "$target"
}

init_conda

if conda env list | awk -v env="$ENV_NAME" '$1 == env { found = 1 } END { exit !found }'; then
    log "Conda environment '$ENV_NAME' already exists; reusing it."
else
    log "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION."
    run conda create "${CONDA_CHANNEL_ARGS[@]}" -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip setuptools wheel
fi

run conda activate "$ENV_NAME"
[[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]] || die "Expected env '$ENV_NAME', got '${CONDA_DEFAULT_ENV:-none}'."

run python -m pip install --upgrade pip setuptools wheel

REQ="$SCRIPT_DIR/requirements_maxwell.txt"
if [[ "$SIMBAV2_CUDA" == "1" ]]; then
    log "Installing dependencies (CUDA 12 JAX)."
    run python -m pip install -r "$REQ"
else
    log "Installing dependencies (CPU-only JAX; skipping jax-cuda12-plugin/pjrt)."
    CPU_REQ="$(mktemp)"
    grep -Ev '^(jax-cuda12-plugin|jax-cuda12-pjrt)==' "$REQ" > "$CPU_REQ"
    run python -m pip install -r "$CPU_REQ"
    rm -f "$CPU_REQ"
fi

log "Installing git dependencies."
clone_or_reuse_repo "cw2" "git@github.com:DongTian95/cw2.git" "dt_branch"

# MetaWorld is installed NON-editable, straight into this env's site-packages,
# rather than shared through $DEPS_DIR. The shared checkout is pinned to
# MetaWorld 2.x because dt_rl's fancy_gym requires it (2.x also pins
# mujoco<3.0.0); this repo needs 3.x, whose "-v3" tasks run the v2 reward
# function by default. Both cannot be one editable checkout, so keep them apart.
log "Installing MetaWorld 3.x into this environment only (not into $DEPS_DIR)."
run python -m pip install --no-cache-dir "git+https://github.com/dongtian-code/Metaworld.git@dt_branch"

# MetaWorld 2.x, if it was ever installed here, pins mujoco<3.0.0 and drags the
# whole env back to 2.3.x. Re-assert the pin after the MetaWorld install.
run python -m pip install "mujoco==3.3.1"

log "Installing this repo (scale_rl) in editable mode."
# --no-deps: setup.py reads deps/requirements.txt, which pins the paper's
# jax 0.4.25 / numpy 1.24 / gymnasium-git stack and would undo everything above.
run python -m pip install -e "$REPO_ROOT" --no-deps

log "Verifying imports."
run python - <<'PY'
import importlib
mods = ["jax", "flax", "optax", "numpy", "gymnasium", "mujoco",
        "metaworld", "hydra", "omegaconf", "wandb", "cw2", "scale_rl"]
for name in mods:
    module = importlib.import_module(name)
    print(f"  {name:12s} {getattr(module, '__version__', 'ok')}")

import jax
print("  jax devices:", jax.devices())

import metaworld, mujoco
assert hasattr(metaworld, "ALL_V3_ENVIRONMENTS"), (
    f"metaworld {metaworld.__version__} at {metaworld.__file__} is 2.x; this repo needs 3.x"
)
assert int(mujoco.__version__.split(".")[0]) >= 3, (
    f"mujoco {mujoco.__version__} is too old for MetaWorld 3.x (a metaworld 2.x "
    "install pins mujoco<3.0.0 and will have downgraded it)"
)

from scale_rl.envs.metaworld import make_metaworld_env, METAWORLD_MT50
print("  metaworld tasks in grid:", len(METAWORLD_MT50))
env = make_metaworld_env(METAWORLD_MT50[0], seed=0)
obs, info = env.reset()
print("  probe:", METAWORLD_MT50[0], "obs", obs.shape,
      "act", env.action_space.shape, "success in info:", "success" in info)
PY

cat <<EOF

[simbav2 setup] Done.

  conda activate $ENV_NAME

Submit the MetaWorld grid with:

  python main_cw2.py configs/cw2/metaworld_online.yml -s

Check the CONDA_PREFIX in configs/cw2/metaworld_online.yml points at this env:

  $(conda info --base)/envs/$ENV_NAME
EOF
