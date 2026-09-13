# SimbaV2

## Introduction

SimbaV2 is a reinforcement learning architecture designed to stabilize training via hyperspherical normalization. By increasing model capacity and compute, SimbaV2 achieves state-of-the-art results on 57 continuous control tasks from MuJoCo, DMControl, MyoSuite, and Humanoid-bench.

<p align="left">
  <img src="docs/images/overview.png" style="width: 50%; max-height: 200px; object-fit: contain;" class="figure">
</p>

<a href="https://joonleesky.github.io" class="nobreak">Hojoon Lee</a><sup>\*</sup>&ensp;
<a href="https://leeyngdo.github.io/" class="nobreak">Youngdo Lee</a><sup>\*</sup>&ensp; 
<a href="https://takuseno.github.io/" class="nobreak">Takuma Seno</a><sup></sup>&ensp;
<a href="https://i-am-proto.github.io" class="nobreak">Donghu Kim</a><sup></sup>&ensp;
<a href="https://www.cs.utexas.edu/~pstone/" class="nobreak">Peter Stone</a><sup></sup>&ensp;
<a href="https://sites.google.com/site/jaegulchoo" class="nobreak">Jaegul Choo</a><sup></sup>&ensp;

[[Website]](https://dojeon-ai.github.io/SimbaV2/) [[Paper]](https://arxiv.org/abs/2502.15280) [[Dataset]](https://dojeon-ai.github.io/SimbaV2/dataset/)

## Result

We compare SimbaV2 to the original Simba by tracking:
- (a) Average normalized return across tasks.
- (b) Weighted sum of $\ell_2$-norms of all intermediate features in critics.
- (c) Weighted sum of $\ell_2$-norms of all critic parameters.
- (d) Weighted sum of $\ell_2$-norms of all gradients in critics.
- (e) Effective learning rate (ELR) of the critics.

SimbaV2 consistently maintains stable norms and ELR, while Simba shows divergent fluctuations.


<p align="center">
  <img src="docs/images/analysis.png" style="width: 95%; object-fit: contain;" class="figure">
</p>

We scale model parameters by increasing critic width and scale compute via the update-to-data (UTD) ratio. We also explore resetting vs. non-resetting training:
- DMC-Hard (7 tasks): $\texttt{dog}$ and $\texttt{humanoid}$ embodiments.
- HBench-Hard (5 tasks): $\texttt{run}$, $\texttt{balance-simple}$, $\texttt{sit-hard}$, $\texttt{stair}$, $\texttt{walk}$.

On these challenging subsets, SimbaV2 benefits from increasing model size and UTD, while Simba plateaus. Notably, SimbaV2 scales smoothly with UTD even without resets, and resetting can degrade its performance.

<p align="center">
  <img src="docs/images/param_scaling.png" style="width: 46%; margin-right: 3%; display:inline-block;" class="figure">
  <img src="docs/images/utd_scaling.png" style="width: 46%; display:inline-block;" class="figure">
</p>

SimbaV2 outperforms competing RL algorithms, with performance improving as compute increases. 

<p align="center">
  <img src="docs/images/online.png" style="width: 95%; object-fit: contain;" class="figure">
</p>


## Getting strated

We use Gymnasium 1.0 API interface which provides seamless integration with diverse RL environments.

### Docker

We provide a `Dockerfile` for easy installation. You can build the docker image by running.

```
docker build . -t scale_rl .
docker run --gpus all -v .:/home/user/scale_rl -it scale_rl /bin/bash
```

### Pip/Conda

If you prefer to install dependencies manually, start by installing dependencies via conda by following the guidelines.
```
# Use pip
pip install -r deps/requirements.txt 

# Or use conda
conda env create -f deps/environment.yaml
```

#### Jax for GPU
```
pip install -U "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# If you want to execute multiple runs with a single GPU, we recommend to set this variable.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

#### Mujoco
Please see installation instruction at [MuJoCo](https://github.com/google-deepmind/mujoco).
```
# Additional environmental evariables for headless rendering
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"
export MKL_SERVICE_FORCE_INTEL="0"
```

#### Humanoid Bench

```
git clone https://github.com/joonleesky/humanoid-bench
cd humanoid-bench
pip install -e .
```

#### Myosuite
```
git clone --recursive https://github.com/joonleesky/myosuite
cd myosuite
pip install -e .
```


##  Example usage

We provide examples on how to train SAC agents with SimBa architecture.  

To run a single online RL experiment
```
python run_online.py
```

To run a single offline RL experiment
```
python run_offline.py
```

To benchmark the algorithm with all environments
```
python run_parallel.py \
    --task all \
    --device_ids <list of gpu devices to use> \
    --num_seeds <num_seeds> \
    --num_exp_per_device <number>  
```


## MetaWorld on SLURM (cw2)

`main_cw2.py` is a second entry point for preemptible cluster runs. It reuses
`run_online.py`'s training loop but drives it from [cw2](https://github.com/ALRhub/cw2),
so a job that Slurm preempts checkpoints itself and resumes exactly where it
stopped after being requeued.

```
bash deps/setup_maxwell.sh                                        # build the conda env
python main_cw2.py configs/cw2/metaworld_online.yml               # run locally, no SLURM
python main_cw2.py configs/cw2/metaworld_online.yml -s            # psgpu: 50 tasks, 1 GPU each
python main_cw2.py configs/cw2/metaworld_online_comgpu.yml -s     # comgpu: 8 tasks on one 4-GPU node
python main_cw2.py configs/cw2/metaworld_online.yml -s -o         # ... overwriting an old submit
```

The two SLURM configs differ in how the partition preempts:

| | `metaworld_online.yml` | `metaworld_online_comgpu.yml` |
|---|---|---|
| partition | psgpu (or allgpu) | comgpu |
| `preemption_mode` | `requeue` — Slurm requeues the job | `cancel` — the job submits its own replacement |
| packing | 1 rep per job | 4 reps per node, one per GPU (`num_gpus: 4`, `reps_per_gpu: 1`) |

In cancel mode a preempted job checkpoints every rep, waits at a checkpoint-ready
barrier under `<path>/.preemption`, and exactly one rep re-`sbatch`es the job's own
`sbatch.sh` with `--array` pinned to this array task. To stop that chain:

```
touch <path>/.preemption/.disable_preemption_resubmit
```

`reps_in_parallel` must equal `num_gpus * reps_per_gpu` or cw2's GPU scheduler
asserts. Raising `reps_per_gpu` above 1 also requires capping JAX's memory
(`XLA_PYTHON_CLIENT_MEM_FRACTION` in `sh_lines`), since JAX otherwise claims 75%
of a GPU on first use and the second rep sharing that device dies on allocation.

What is different from `run_online.py`:

- **Config.** The cw2 YAML's `params` block is translated into Hydra overrides
  and composed against this repo's own `configs/` tree, so `configs/online_rl.yaml`
  still owns every derived value. `defaults:` selects config groups
  (`env: metaworld`), everything else becomes a dotted override.
- **Checkpoints.** `scale_rl/common/checkpoint.py` writes parameters, optimizer
  state, the agent PRNG key, the normalizer statistics and the whole replay
  buffer as one atomic pickle -- unlike `BaseAgent.save()`, which stores weights
  only and is meant for transfer, not resume.
- **Environments.** `configs/env/metaworld.yaml` + `scale_rl/envs/metaworld.py`
  add MetaWorld 3.x's 50 ML1 tasks (goal-observable, 39-D observations, 500-step
  episodes, no early termination).
- **Dependencies.** MetaWorld 3.x needs gymnasium >= 1.1 and mujoco 3.3, which
  `deps/requirements.txt` (the paper's pinned stack) cannot provide. Use
  `deps/requirements_maxwell.txt` for these runs.

On resume the environment is re-`reset()` rather than restored to its exact
mid-episode simulator state, so only the episode that was in flight is lost.


## Analysis

Please refer to `/analysis` to visualize the experimental results provided in the paper.


## License
This project is released under the [Apache 2.0 license](/LICENSE).

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@article{lee2025hyperspherical,
  title={Hyperspherical Normalization for Scalable Deep Reinforcement Learning},
  author={Lee, Hojoon and Lee, Youngdo and Seno, Takuma and Kim, Donghu and Stone, Peter and Choo, Jaegul},
  journal={arXiv preprint arXiv:2502.15280},
  year={2025}
}
```
