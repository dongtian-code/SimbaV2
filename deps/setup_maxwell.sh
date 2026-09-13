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

# Clone (or update) a git dependency next to the repo and pip install -e it.
# Editable installs keep the source readable on the cluster, which matters when
# a job's traceback points into cw2 or MetaWorld.
clone_or_update_repo() {
    local name="$1" url="$2" branch="$3" target="$DEPS_DIR/$name"
    mkdir -p "$DEPS_DIR"
    if [[ -d "$target/.git" ]]; then
        log "Updating $name in $target."
        run git -C "$target" fetch --all --prune
        run git -C "$target" checkout "$branch"
        run git -C "$target" pull --ff-only || log "Could not fast-forward $name; leaving it as is."
    else
        log "Cloning $name into $target."
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
clone_or_update_repo "cw2"       "git@github.com:DongTian95/cw2.git"          "dt_branch"
clone_or_update_repo "Metaworld" "git@github.com:dongtian-code/Metaworld.git" "dt_branch"

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

from scale_rl.envs.metaworld import metaworld_task_name, METAWORLD_MT50
print("  metaworld tasks in grid:", len(METAWORLD_MT50),
      "->", metaworld_task_name(METAWORLD_MT50[0]))
PY

cat <<EOF

[simbav2 setup] Done.

  conda activate $ENV_NAME

Submit the MetaWorld grid with:

  python main_cw2.py configs/cw2/metaworld_online.yml -s

Check the CONDA_PREFIX in configs/cw2/metaworld_online.yml points at this env:

  $(conda info --base)/envs/$ENV_NAME
EOF
