"""cw2 entry point for SimbaV2, with SLURM (requeue) auto-resume support.

This is `run_online.py`'s training loop restructured as a cw2
`AbstractIterativeExperiment`: `iterate()` advances the run by `chunk_size`
interaction steps and cw2 calls `save_state()` after each one, which is what
gives a preempted job a checkpoint to come back to.

It mirrors the scheme already used by the sibling RLAC repo, so both projects
behave identically on the cluster:
  - the same config keys: `auto_resume_from_latest_checkpoint`, `resume_dir_name`,
    `resume_scope_name`, `preemption_mode`, `num_checkpoints`,
    `overwrite_checkpoints`, `strict_checkpoint_config`, `active_run_lock`.
  - both preemption paths. On `allgpu` (`preemption_mode: requeue`) Slurm puts
    the job back in the queue itself, so the run checkpoints and then quiesces at
    that exact boundary. On `comgpu`/`compgpu` (`preemption_mode: cancel`) Slurm
    just kills the job, so the run has to submit its own replacement: every rep
    packed into the Slurm job checkpoints, they meet at a checkpoint-ready
    barrier, exactly one of them `sbatch`es the original job script again, and
    then they all exit.

cw2 itself has no restore/resume hooks of any kind -- `AbstractIterativeExperiment`
only calls `save_state()` after every `iterate()`. Everything below (checkpoint
discovery, the active-run lock, signal handling, requeue quiescing) is
application-level code living in `initialize()`/`iterate()`.

Config flow: the cw2 YAML's `params` block is translated into Hydra overrides and
composed against the repo's own `configs/` tree, so `configs/online_rl.yaml` keeps
owning the derived values (`gamma` from the episode length, the learning-rate
decay horizon, the layer scaler inits) instead of them being duplicated here.

Known scope simplification: on resume the environment is re-`reset()` rather than
restored to its exact mid-episode simulator state. Networks, optimizer state,
normalizer statistics, replay buffer, RNG streams and step counters all come back
exactly; only the position within the episode that was in flight is lost.
"""

import errno
import fcntl
import glob
import hashlib
import json
import math  # noqa: F401 -- in scope for the configs' ${eval:'math.sqrt(...)'}
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

import hydra
import numpy as np
import omegaconf
from cw2 import cluster_work, cw_error, experiment
from cw2.cw_config import cw_config as cw_config_module
from cw2.cw_data import cw_logging
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.common.checkpoint import (
    META_SUFFIX,
    load_checkpoint,
    read_checkpoint_extra,
    save_checkpoint,
)
from scale_rl.envs import create_envs
from scale_rl.evaluation import evaluate, record_video


# --------------------------------------------------------------------------- #
# cw2 config helpers (kept byte-compatible with RLAC's main_cw2.py so a
# checkpoint tree laid out by one repo is readable by the other's conventions)
# --------------------------------------------------------------------------- #
def _bool(value):
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _safe_path_component(value):
    raw = str(value).strip()
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', raw).strip('._-') or 'default'
    return f'{safe[:80]}_{digest}'


def _resume_scope_name(cw_config):
    scope = cw_config.get('resume_scope_name')
    if scope is None:
        scope = cw_config.get('sub_exp_name')
    if scope is None and isinstance(cw_config.get('wandb'), dict):
        scope = cw_config['wandb'].get('group')
    if scope is None:
        return None
    return _safe_path_component(scope)


def _checkpoint_config_fingerprint(cw_config):
    payload = {
        'iterations': cw_config.get('iterations'),
        'seed': cw_config.get('seed'),
        'params': cw_config.get('params'),
        # The resume scope (explicit `resume_scope_name`/`sub_exp_name`, else the
        # wandb group) is part of the identity of a checkpoint lineage. Without it
        # a run relaunched under a new group would silently adopt the old group's
        # checkpoints.
        'resume_scope': _resume_scope_name(cw_config),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=repr)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


# Slurm partitions differ in what preemption does. On `allgpu` a preempted job is
# requeued by Slurm; on `comgpu`/`compgpu` it is simply cancelled and nothing
# comes back unless the job submits a replacement itself.
_REQUEUE_MODES = frozenset({'requeue', 'allgpu'})
_CANCEL_MODES = frozenset({'cancel', 'comgpu', 'compgpu'})

# Keys that belong to the preemption scheme but are natural to write in the SLURM
# document (next to `partition`). `prepare_rep_configs` copies them down into each
# repetition config, which is all `initialize()`/`iterate()` ever sees.
_SLURM_PREEMPTION_KEYS = (
    'partition',
    'preemption_mode',
    'auto_resume_from_latest_checkpoint',
    'resume_dir_name',
    'disable_preemption_resubmit',
    'exclude_current_node_on_resubmit',
    'checkpoint_ready_barrier_enabled',
    'checkpoint_ready_barrier_timeout',
    'checkpoint_ready_barrier_poll_interval',
    'checkpoint_ready_barrier_submit_on_timeout',
)


def _get_preemption_mode(cw_config):
    mode = str(cw_config.get('preemption_mode', '') or '').lower()
    if mode:
        return mode
    partition = str(cw_config.get('partition', '') or '').lower()
    if partition == 'allgpu':
        return 'requeue'
    if partition in {'comgpu', 'compgpu'}:
        return 'cancel'
    return ''


def _slurm_marker_job_id():
    """Identify the Slurm allocation the current process belongs to.

    Array jobs need both ids: every array task of one submission shares
    SLURM_ARRAY_JOB_ID, and a marker keyed on that alone would let one task's
    replacement suppress every other task's.
    """
    array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
    array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_job_id is not None and array_task_id is not None:
        return f'{array_job_id}_{array_task_id}'
    return os.environ.get('SLURM_JOB_ID', 'local')


def _align_egl_device_with_cuda():
    """Point MuJoCo's EGL backend at the GPU this process was actually given.

    Has to run per repetition rather than once in `__main__`: cw2's
    GPU-distributing scheduler assigns CUDA_VISIBLE_DEVICES *inside* each worker
    process, long after `__main__` finished. MuJoCo's EGL backend also takes a
    single device index, so a whole-node "0,1,2,3" must be narrowed to one entry.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    if visible.isdigit():
        os.environ['EGL_DEVICE_ID'] = visible
        os.environ['MUJOCO_EGL_DEVICE_ID'] = visible


def _replace_path_component(path, predicate, new_component):
    parts = os.path.normpath(path).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if predicate(parts[i]):
            parts[i] = new_component
            break
    else:
        parts.append(new_component)
    if os.path.isabs(path):
        return os.sep + os.path.join(*[p for p in parts if p])
    return os.path.join(*parts)


def prepare_rep_configs(config_obj: cw_config_module.Config):
    """Resolve seed/resume/timestamp paths for every unfolded repetition.

    Must run after cw2 unfolds `exp_configs` (i.e. after `ClusterWork(...)` is
    constructed) and before `cw.run()`, which creates the on-disk directories and
    re-dumps the resolved YAML.
    """
    slurm_config = getattr(config_obj, 'slurm_config', None) or {}
    for rep_config in config_obj.exp_configs:
        # cw2 keeps the SLURM document separate from the experiment documents, and
        # only the latter reach the job. Copy the preemption keys across so they
        # can be written next to `partition`, where they belong.
        for key in _SLURM_PREEMPTION_KEYS:
            if key in slurm_config and key not in rep_config:
                rep_config[key] = slurm_config[key]

        base_seed = int(rep_config.get('seed', 0))
        rep_idx = int(rep_config.get('_rep_idx', 0))
        seed = base_seed + rep_idx
        rep_config['seed'] = seed

        # cw2 unfolds repetitions as rep_00, rep_01, ... -- use the concrete seed
        # as the persistent repetition identity instead, so a later supplemental
        # seed launch cannot collide with an earlier checkpoint.
        rep_config['_rep_log_path'] = _replace_path_component(
            rep_config['_rep_log_path'],
            lambda p: re.fullmatch(r'rep_-?\d+', p) is not None,
            f'rep_{seed:02d}',
        )

        auto_resume = _bool(rep_config.get('auto_resume_from_latest_checkpoint', False))
        if auto_resume:
            resume_dir_name = rep_config.get('resume_dir_name', 'resume')
            basic_path = os.path.abspath(rep_config.get('path'))
            resume_root = os.path.join(basic_path, resume_dir_name)
            scope = _resume_scope_name(rep_config)
            if scope is not None:
                rep_config['resume_scope_name'] = scope
                resume_root = os.path.join(resume_root, scope)
            rep_config['resume_model_dir'] = os.path.join(resume_root, f'rep_{seed:02d}', 'model')
            # Infer from `partition` rather than defaulting to requeue: on a
            # cancel-mode partition a requeue-mode job would checkpoint, quiesce,
            # and then be killed with nothing scheduled to take its place.
            inferred = _get_preemption_mode(rep_config)
            rep_config['preemption_mode'] = inferred or 'requeue'

        # Every run/requeue segment gets its own timestamped scratch log dir. The
        # resume dir above is untouched by the timestamp -- that is what makes
        # checkpoints discoverable across restarts.
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        for key in ('log_path', '_rep_log_path'):
            rep_config[key] = os.path.abspath(_replace_path_component(
                rep_config[key],
                lambda p: p == 'log' or p.startswith('log_'),
                f'log_{timestamp}',
            ))


# --------------------------------------------------------------------------- #
# cw2 params -> Hydra overrides
# --------------------------------------------------------------------------- #
# Keys under `params` that configure the launcher itself and must not be handed
# to Hydra as config overrides.
_CW2_ONLY_PARAM_KEYS = frozenset({'config_path', 'config_name', 'defaults', 'chunk_size', 'seed'})


def _override_literal(value):
    """Render a Python value in Hydra's override grammar."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(_override_literal(v) for v in value) + ']'
    # Quote every string: task ids like "metaworld/reach-v2" and group names with
    # spaces or brackets are not valid unquoted override values.
    escaped = str(value).replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


def _flatten_params(node, prefix=''):
    for key, value in node.items():
        path = f'{prefix}.{key}' if prefix else str(key)
        if not prefix and key in _CW2_ONLY_PARAM_KEYS:
            continue
        if isinstance(value, dict):
            yield from _flatten_params(value, path)
        else:
            yield path, value


def _hydra_overrides(params, seed):
    overrides = []
    # Defaults-list selections ("env=metaworld") have to be expressed as group
    # overrides, not as dotted value overrides.
    for group, option in (params.get('defaults') or {}).items():
        overrides.append(f'{group}={option}')
    # cw2 owns the seed: `prepare_rep_configs` derives it per repetition.
    overrides.append(f'seed={int(seed)}')
    for path, value in _flatten_params(params):
        overrides.append(f'{path}={_override_literal(value)}')
    return overrides


def _compose_cfg(cw_config):
    """Compose the repo's own Hydra config, overridden by the cw2 `params` block."""
    params = dict(cw_config.get('params') or {})
    config_name = params.get('config_name', 'online_rl')
    overrides = _hydra_overrides(params, cw_config['seed'])

    # `initialize_config_dir` (absolute) rather than `initialize` (relative):
    # `hydra.initialize` resolves a relative config_path against the *caller's*
    # location, falling back to the working directory -- and a SLURM job does not
    # control its working directory. Anchoring on this file instead also means the
    # configs are read out of cw2's code copy, i.e. the snapshot this job was
    # submitted with, rather than whatever the source tree looks like now.
    config_dir = params.get('config_path', 'configs')
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_dir)
    config_dir = os.path.abspath(config_dir)

    def eval_resolver(s: str):
        return eval(s)

    # replace=True: cw2 may construct more than one experiment object in a
    # process, and registering the resolver twice is an error.
    omegaconf.OmegaConf.register_new_resolver('eval', eval_resolver, replace=True)

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    omegaconf.OmegaConf.resolve(cfg)

    print(f'[config] Hydra config dir: {config_dir}', flush=True)

    print('[config] Hydra overrides: ' + ' '.join(overrides), flush=True)
    return cfg


class SimbaV2Experiment(experiment.AbstractIterativeExperiment):
    checkpoint_pattern = re.compile(r'^checkpoint_state_(\d+)$')
    latest_checkpoint_name = 'checkpoint_state'

    @classmethod
    def _is_checkpoint_entry(cls, name):
        """True for a checkpoint file or its `.meta` sidecar."""
        base = name[: -len(META_SUFFIX)] if name.endswith(META_SUFFIX) else name
        return base == cls.latest_checkpoint_name or bool(cls.checkpoint_pattern.match(base))

    # ---------------------------------------------------------------- checkpoint discovery
    @classmethod
    def _checkpoint_metadata(cls, checkpoint_path, fallback_epoch=None):
        extra = read_checkpoint_extra(checkpoint_path)
        if extra is None:
            return None
        epoch = extra.get('num_iterations', fallback_epoch)
        if epoch is None:
            return None
        try:
            epoch = int(epoch)
        except (TypeError, ValueError):
            return None
        return epoch, extra.get('config_fingerprint')

    @classmethod
    def _latest_checkpoint_info(cls, model_dir):
        if model_dir is None or not os.path.isdir(model_dir):
            return None, None
        candidates = []
        latest_path = os.path.join(model_dir, cls.latest_checkpoint_name)
        if os.path.isfile(latest_path):
            metadata = cls._checkpoint_metadata(latest_path)
            if metadata is not None:
                epoch, fingerprint = metadata
                candidates.append((epoch, os.path.getmtime(latest_path), fingerprint))

        suffixed = []
        for name in os.listdir(model_dir):
            m = cls.checkpoint_pattern.match(name)
            if m:
                suffixed.append((int(m.group(1)), os.path.join(model_dir, name)))
        for epoch_guess, path in sorted(suffixed, reverse=True):
            metadata = cls._checkpoint_metadata(path, fallback_epoch=epoch_guess)
            if metadata is None:
                continue
            epoch, fingerprint = metadata
            candidates.append((epoch, os.path.getmtime(path), fingerprint))
            break  # file names encode the epoch; the first readable one is newest

        if not candidates:
            return None, None
        epoch, _, fingerprint = max(candidates, key=lambda item: item[:2])
        return epoch, fingerprint

    @classmethod
    def _checkpoint_mtime(cls, model_dir):
        candidates = [os.path.join(model_dir, cls.latest_checkpoint_name)]
        if os.path.isdir(model_dir):
            for name in os.listdir(model_dir):
                if cls.checkpoint_pattern.match(name):
                    candidates.append(os.path.join(model_dir, name))
        mtimes = [os.path.getmtime(p) for p in candidates if os.path.isfile(p)]
        return max(mtimes) if mtimes else 0.0

    @classmethod
    def _rep_dir_names(cls, cw_config):
        names = []
        for value in (cw_config.get('seed'), cw_config.get('_rep_idx')):
            if value is None:
                continue
            try:
                v = int(value)
            except (TypeError, ValueError):
                continue
            for cand in (f'rep_{v:02d}', f'rep_{v}'):
                if cand not in names:
                    names.append(cand)
        return names

    @classmethod
    def _group_roots_for_checkpoint_search(cls, cw_config, resume_model_dir):
        roots = []
        for key in ('path', '_basic_path'):
            p = cw_config.get(key)
            if p is not None:
                roots.append(os.path.abspath(p))
        for key in ('log_path', '_rep_log_path', 'save_model_dir', 'resume_model_dir'):
            p = cw_config.get(key)
            if p is None:
                continue
            p = os.path.abspath(p)
            parts = os.path.normpath(p).split(os.sep)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == 'resume' or parts[i] == 'log' or parts[i].startswith('log_'):
                    roots.append(os.sep + os.path.join(*[x for x in parts[:i] if x]))
                    break
        if resume_model_dir is not None:
            d = os.path.abspath(resume_model_dir)
            for _ in range(5):
                roots.append(d)
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        seen = set()
        unique = []
        for r in roots:
            r = os.path.abspath(r)
            if r not in seen and os.path.isdir(r):
                unique.append(r)
                seen.add(r)
        return unique

    @classmethod
    def _candidate_checkpoint_dirs(cls, cw_config, resume_model_dir):
        candidates = []
        if resume_model_dir is not None:
            candidates.append(os.path.abspath(resume_model_dir))
        rep_dir_names = cls._rep_dir_names(cw_config)
        scope = _resume_scope_name(cw_config)
        for root in cls._group_roots_for_checkpoint_search(cw_config, resume_model_dir):
            resume_dir = os.path.join(root, 'resume')
            for rep_dir_name in rep_dir_names:
                if scope is not None:
                    candidates.append(os.path.join(resume_dir, scope, rep_dir_name, 'model'))
                else:
                    candidates.append(os.path.join(resume_dir, rep_dir_name, 'model'))
            # Timestamped run dirs are only searched for an unscoped lineage: they
            # carry no scope of their own, so scanning them under a scope would
            # pull in checkpoints from every other group that ran this grid point.
            if scope is not None:
                continue
            for run_dir in [os.path.join(root, 'log'), *glob.glob(os.path.join(root, 'log_*'))]:
                for rep_dir_name in rep_dir_names:
                    candidates.append(os.path.join(run_dir, rep_dir_name, 'model'))
        seen = set()
        unique = []
        for c in candidates:
            c = os.path.abspath(c)
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    @classmethod
    def _find_latest_checkpoint(cls, cw_config, resume_model_dir):
        """Return (dir, epoch, None, None) if resumable, or (None, None, dir, epoch)
        if the matching run is already complete, or (None, None, None, None)."""
        best = None
        completed = None
        max_epoch = cw_config.get('iterations')
        if max_epoch is not None:
            max_epoch = int(max_epoch)
        strict = _bool(cw_config.get('strict_checkpoint_config', True))
        expected_fp = _checkpoint_config_fingerprint(cw_config) if strict else None
        for model_dir in cls._candidate_checkpoint_dirs(cw_config, resume_model_dir):
            epoch, fingerprint = cls._latest_checkpoint_info(model_dir)
            if epoch is None:
                continue
            if strict and fingerprint is not None and fingerprint != expected_fp:
                print(f'[checkpoint] Ignoring checkpoint with mismatched config fingerprint: {model_dir}', flush=True)
                continue
            key = (epoch, cls._checkpoint_mtime(model_dir))
            if max_epoch is not None and epoch >= max_epoch:
                if completed is None or key > completed[0]:
                    completed = (key, model_dir, epoch)
                continue
            if best is None or key > best[0]:
                best = (key, model_dir, epoch)
        if completed is not None:
            _, model_dir, epoch = completed
            return None, None, model_dir, epoch
        if best is None:
            return None, None, None, None
        _, model_dir, epoch = best
        return model_dir, epoch, None, None

    # ---------------------------------------------------------------- active-run lock
    def _active_run_lock_enabled(self, cw_config):
        if _bool(cw_config.get('disable_active_run_lock', False)):
            return False
        return _bool(cw_config.get(
            'active_run_lock', cw_config.get('auto_resume_from_latest_checkpoint', False)
        ))

    def _active_run_lock_path(self, cw_config):
        if self.resume_model_dir is not None:
            return os.path.join(os.path.dirname(self.resume_model_dir), 'active.lock')
        save_model_dir = getattr(self, 'save_model_dir', None)
        if save_model_dir is not None:
            return os.path.join(os.path.dirname(os.path.abspath(save_model_dir)), 'active.lock')
        return None

    def _acquire_active_run_lock(self, cw_config):
        lock_path = self._active_run_lock_path(cw_config)
        if lock_path is None:
            return True
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                raise
            with os.fdopen(fd, 'r') as f:
                holder = f.read().strip()
            print(f'[checkpoint] Active run lock held at {lock_path}; skipping duplicate run. Holder: {holder}', flush=True)
            return False
        metadata = dict(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
            slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
            started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(metadata, sort_keys=True).encode('utf-8'))
        os.fsync(fd)
        self._active_run_lock_fd = fd
        return True

    def _release_active_run_lock(self):
        fd = getattr(self, '_active_run_lock_fd', None)
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            self._active_run_lock_fd = None

    # ---------------------------------------------------------------- preemption (requeue only)
    def _register_preemption_handlers(self):
        def _request(signum, _frame):
            self._preemption_requested = True
            self._preemption_signal = signum
            print(f'[preemption] Received signal {signum}; will checkpoint after the current step.', flush=True)

        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(sig, _request)
            except (AttributeError, ValueError):
                pass

    def _wait_for_requeue_termination(self):
        """Sit exactly at the USR1 checkpoint boundary until Slurm SIGTERMs us."""
        while self._preemption_signal != signal.SIGTERM:
            signal.pause()
        print('[preemption] Termination received while quiesced; exiting at the saved checkpoint boundary.', flush=True)
        raise cw_error.ExperimentSurrender()

    def _handle_preemption(self, cw_config):
        checkpoint_path = None
        try:
            checkpoint_path = self._save_checkpoint(cw_config, force=True)
        except Exception as error:
            # Keep going: the last good checkpoint is still on disk, and in
            # cancel mode a replacement job is worth submitting even if this
            # write failed -- it will resume from the previous checkpoint.
            print(f'[preemption] Checkpoint save failed: {error}', flush=True)
        self._skip_next_save_state = True

        preemption_mode = _get_preemption_mode(cw_config) or 'requeue'

        if preemption_mode in _CANCEL_MODES:
            # Slurm will not bring this job back, so announce the checkpoint,
            # wait for the other reps in this job, and submit the replacement.
            # This runs for SIGTERM too: on a cancel-mode partition SIGTERM is
            # how the preemption finishes, not a sign that the user cancelled.
            #
            # Drop the active-run lock *before* publishing readiness. The rep
            # that submits only gets past the barrier once every rep has
            # published, so this ordering guarantees no rep is still holding its
            # lock when the replacement job starts -- a rep whose lock is still
            # held would be skipped by the replacement and silently lost.
            # Training is already over at this point; the checkpoint is written.
            self._release_active_run_lock()
            self._publish_checkpoint_ready(cw_config, checkpoint_path)
            self._resubmit_if_cancel_preemption(cw_config)
            print('[preemption] Cancel-mode handling complete; exiting.', flush=True)
            raise cw_error.ExperimentSurrender({'preempted': True, 'preemption_mode': preemption_mode})

        if self._preemption_signal == signal.SIGTERM:
            print('[preemption] Termination signal checkpoint complete; exiting.', flush=True)
            raise cw_error.ExperimentSurrender()

        if preemption_mode in _REQUEUE_MODES:
            print(
                '[preemption] Requeue-mode checkpoint complete; quiescing until Slurm '
                'requeues or terminates the job.', flush=True,
            )
            self._wait_for_requeue_termination()

        print('[preemption] Checkpoint complete; no preemption_mode configured, continuing.', flush=True)
        self._preemption_requested = False

    # ------------------------------------------------- cancel-mode resubmission
    # On a cancel-mode partition Slurm does not bring a preempted job back, so
    # the job submits its own replacement. Two things have to be coordinated
    # across the reps packed into one Slurm job (`reps_in_parallel`), each of
    # which is a separate process that gets its own SIGUSR1:
    #
    #   1. the barrier -- no replacement may be submitted until *every* rep in
    #      the job has a fresh checkpoint on disk. Otherwise the replacement can
    #      start while a rep is still writing, and the active-run lock would make
    #      the new job skip that rep entirely, silently dropping it.
    #   2. the claim -- exactly one rep may call sbatch, or one preemption turns
    #      into `reps_in_parallel` replacement jobs.
    #
    # Both are files under `<path>/.preemption`, keyed by the Slurm job id.
    def _marker_dir(self, cw_config):
        """A directory shared by every rep in this Slurm job.

        Not `path`: cw2 extends it per grid point, and one Slurm job packs
        several grid points. `_basic_path` is the un-extended root and is
        identical for every rep.
        """
        for key in ('_basic_path', 'path'):
            base = cw_config.get(key)
            if base:
                return os.path.join(os.path.abspath(base), '.preemption')
        return os.path.join(os.getcwd(), '.preemption')

    def _bool_cfg(self, cw_config, key, default):
        return _bool(cw_config.get(key, default))

    def _resubmit_disabled(self, cw_config):
        if os.environ.get('SIMBAV2_DISABLE_PREEMPTION_RESUBMIT', '').lower() in {'1', 'true', 'yes', 'on'}:
            return True
        if self._bool_cfg(cw_config, 'disable_preemption_resubmit', False):
            return True
        # A drop file is the way to stop a self-perpetuating chain of jobs without
        # editing the config or waiting for the current ones to finish.
        marker_dir = self._marker_dir(cw_config)
        job_id = _slurm_marker_job_id()
        for name in ('.disable_preemption_resubmit', f'.disable_preemption_resubmit_{job_id}'):
            if os.path.exists(os.path.join(marker_dir, name)):
                print(f'[preemption] Found {name}; not submitting a replacement.', flush=True)
                return True
        return False

    # --- checkpoint-ready barrier ---
    @staticmethod
    def _barrier_task_id(value):
        return str(int(value)) if isinstance(value, (int, np.integer)) else str(value)

    def _barrier_current_task_id(self, cw_config):
        task_id = cw_config.get('_cw2_job_task_id')
        if task_id is None:
            task_id = cw_config.get('_rep_idx', cw_config.get('seed'))
        if task_id is None:
            raise RuntimeError('checkpoint barrier cannot identify the current task')
        return self._barrier_task_id(task_id)

    def _barrier_expected_task_ids(self, cw_config):
        raw = cw_config.get('_cw2_job_task_ids')
        if raw is None:
            # Repetition ids are not unique once a Slurm task packs several sweep
            # points, so only a single-task job may fall back to them.
            count = int(cw_config.get('_cw2_job_task_count', cw_config.get('reps_per_job', 1)) or 1)
            if count != 1:
                raise RuntimeError(
                    'checkpoint barrier needs _cw2_job_task_ids from cw2 for a job '
                    f'packing {count} runs; update the cw2 checkout'
                )
            return [self._barrier_current_task_id(cw_config)]
        task_ids = [self._barrier_task_id(t) for t in raw]
        if len(set(task_ids)) != len(task_ids):
            raise RuntimeError(f'checkpoint barrier got duplicate task ids: {task_ids}')
        current = self._barrier_current_task_id(cw_config)
        if current not in task_ids:
            raise RuntimeError(
                f'checkpoint barrier: current task {current} is not in the job membership {task_ids}'
            )
        return task_ids

    def _barrier_marker_path(self, cw_config, task_id):
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', self._barrier_task_id(task_id))
        return os.path.join(
            self._marker_dir(cw_config),
            f'.checkpoint_ready_{_slurm_marker_job_id()}',
            f'task_{safe[:48]}.json',
        )

    @staticmethod
    def _atomic_write_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp.{os.getpid()}.{time.time_ns()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _publish_checkpoint_ready(self, cw_config, checkpoint_path, status='ready'):
        """Announce that this rep is done with the node.

        `status='ready'` means "checkpoint written, safe to be replaced".
        `status='skipped'` means this rep will not run at all in this job -- it is
        already complete, or another process holds its lock. Publishing that is
        what keeps one skipped rep from stalling the barrier for the whole job,
        which would leave the remaining reps with no replacement submitted.
        """
        if _get_preemption_mode(cw_config) not in _CANCEL_MODES:
            return
        if not self._bool_cfg(cw_config, 'checkpoint_ready_barrier_enabled', True):
            return
        try:
            task_id = self._barrier_current_task_id(cw_config)
        except RuntimeError as error:
            print(f'[checkpoint barrier] Cannot publish readiness: {error}', flush=True)
            return
        payload = {
            'task_id': task_id,
            'job_id': _slurm_marker_job_id(),
            'status': status,
            'checkpoint_path': os.path.abspath(checkpoint_path) if checkpoint_path else None,
            'n_completed': getattr(self, 'n_completed', None),
            'interaction_step': getattr(self, 'interaction_step', None),
            'seed': cw_config.get('seed'),
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = self._barrier_marker_path(cw_config, task_id)
        self._atomic_write_json(path, payload)
        print(f'[checkpoint barrier] task {task_id} {status}: {path}', flush=True)

    def _barrier_ready_task_ids(self, cw_config, task_ids):
        ready = set()
        job_id = _slurm_marker_job_id()
        for task_id in task_ids:
            try:
                with open(self._barrier_marker_path(cw_config, task_id), encoding='utf-8') as f:
                    marker = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            # Reject a marker left behind by an earlier allocation, and one whose
            # checkpoint has since disappeared.
            status = marker.get('status')
            if status not in {'ready', 'skipped'} or str(marker.get('job_id')) != job_id:
                continue
            if status == 'ready':
                checkpoint_path = marker.get('checkpoint_path')
                if checkpoint_path is None or not os.path.exists(checkpoint_path):
                    continue
            ready.add(self._barrier_task_id(marker.get('task_id')))
        return ready

    def _wait_for_checkpoint_ready_barrier(self, cw_config):
        if not self._bool_cfg(cw_config, 'checkpoint_ready_barrier_enabled', True):
            return True
        try:
            task_ids = self._barrier_expected_task_ids(cw_config)
        except RuntimeError as error:
            print(f'[checkpoint barrier] {error}; not resubmitting.', flush=True)
            return False

        timeout = float(cw_config.get('checkpoint_ready_barrier_timeout', 240.0))
        poll = float(cw_config.get('checkpoint_ready_barrier_poll_interval', 0.5))
        if timeout < 0 or poll <= 0:
            print('[checkpoint barrier] Invalid timeout/poll interval; not resubmitting.', flush=True)
            return False

        deadline = time.monotonic() + timeout
        last_missing = None
        while True:
            ready = self._barrier_ready_task_ids(cw_config, task_ids)
            missing = [t for t in task_ids if t not in ready]
            if not missing:
                print(f'[checkpoint barrier] All {len(task_ids)} run(s) in this job are ready.', flush=True)
                return True
            if missing != last_missing:
                print(f'[checkpoint barrier] Waiting for task(s) {missing}; ready={sorted(ready)}.', flush=True)
                last_missing = missing
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                submit_anyway = self._bool_cfg(cw_config, 'checkpoint_ready_barrier_submit_on_timeout', False)
                print(
                    f'[checkpoint barrier] Timed out after {timeout:.1f}s; missing {missing}; '
                    f'{"submitting anyway" if submit_anyway else "not resubmitting"}.',
                    flush=True,
                )
                return submit_anyway
            time.sleep(min(poll, remaining))

    # --- the replacement command ---
    @staticmethod
    def _path_has_component(path, predicate):
        return any(predicate(part) for part in os.path.normpath(path).split(os.sep))

    def _find_original_sbatch_script(self, cw_config):
        """Locate the sbatch.sh cw2 generated for this submission.

        It sits in the code copy this job is running out of, so walking up from
        this file finds the script that produced exactly this job -- the same
        grid, the same resources, the same code snapshot.
        """
        search_dirs = []
        for base in (os.path.abspath(sys.argv[0]), os.path.abspath(__file__), os.getcwd()):
            d = base if os.path.isdir(base) else os.path.dirname(base)
            for _ in range(12):
                search_dirs.append(d)
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        for d in search_dirs:
            candidate = os.path.join(d, 'sbatch.sh')
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def _replacement_exclude_nodes(self, cw_config):
        if not self._bool_cfg(cw_config, 'exclude_current_node_on_resubmit', False):
            return None
        if os.environ.get('SLURM_JOB_ID') is None:
            return None
        node_list = (
            os.environ.get('SLURM_JOB_NODELIST')
            or os.environ.get('SLURM_NODELIST')
            or socket.gethostname().split('.', 1)[0]
        ).strip()
        # Goes straight onto an sbatch command line, so only accept a Slurm
        # hostlist ("max-wng023" / "max-wng[023-025]").
        if not re.fullmatch(r'[A-Za-z0-9_.\-\[\],]+', node_list):
            print(f'[resubmit] Ignoring malformed Slurm node list: {node_list!r}', flush=True)
            return None
        return node_list

    def _resubmit_command(self, cw_config):
        sbatch_script = self._find_original_sbatch_script(cw_config)
        if sbatch_script is None:
            print(
                '[resubmit] Could not find the generated sbatch.sh next to this code copy; '
                'falling back to re-running the launcher with --nocodecopy.', flush=True,
            )
            args = [os.path.abspath(sys.argv[0]), *self._strip_job_args(sys.argv[1:])]
            for flag, aliases in (('-s', ('--slurm',)), ('-o', ('--overwrite',))):
                if flag not in args and not any(a in args for a in aliases):
                    args.append(flag)
            if '--nocodecopy' not in args:
                args.append('--nocodecopy')
            return ' '.join(shlex.quote(part) for part in [sys.executable, *args])

        command = ['sbatch']
        exclude_nodes = self._replacement_exclude_nodes(cw_config)
        if exclude_nodes is not None:
            command.append(f'--exclude={exclude_nodes}')
            print(f'[resubmit] Excluding the current node(s) from the replacement: {exclude_nodes}', flush=True)
        array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if array_task_id is not None:
            # Resubmit only this array task. Without it the whole array comes
            # back, and every other task's reps would be skipped by the
            # active-run lock or, worse, restarted from their own checkpoints.
            command.append(f'--array={array_task_id}')
        command.append(sbatch_script)
        return ' '.join(shlex.quote(part) for part in command)

    @staticmethod
    def _strip_job_args(args):
        """Drop cw2's `-j/--job <idx>`, which pins the rerun to one job index."""
        stripped = []
        skip = False
        for arg in args:
            if skip:
                skip = False
                continue
            if arg in {'-j', '--job'}:
                skip = True
                continue
            if arg.startswith('--job='):
                continue
            stripped.append(arg)
        return stripped

    def _claim_resubmission(self, cw_config):
        marker_dir = self._marker_dir(cw_config)
        os.makedirs(marker_dir, exist_ok=True)
        marker_path = os.path.join(marker_dir, f'.preemption_resubmitted_{_slurm_marker_job_id()}')
        try:
            # O_EXCL is the whole mechanism: the first rep to get here wins and
            # every other rep in the job sees FileExistsError.
            fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'status': 'claimed',
                       'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, sort_keys=True)
        return marker_path

    def _resubmit_if_cancel_preemption(self, cw_config):
        if _get_preemption_mode(cw_config) not in _CANCEL_MODES:
            return False
        if self._resubmit_disabled(cw_config):
            print('[preemption] Replacement submission disabled; checkpoint only.', flush=True)
            return False
        if not self._wait_for_checkpoint_ready_barrier(cw_config):
            print('[preemption] Checkpoint-ready barrier not satisfied; not resubmitting.', flush=True)
            return False

        command = self._resubmit_command(cw_config)
        marker_path = self._claim_resubmission(cw_config)
        if marker_path is None:
            print('[preemption] A replacement job was already submitted by another rep in this job.', flush=True)
            return False

        print(f'[preemption] Submitting replacement job: {command}', flush=True)
        result = subprocess.run(command, shell=True, cwd=os.getcwd())
        with open(marker_path, 'w') as f:
            json.dump({'pid': os.getpid(), 'command': command, 'returncode': result.returncode,
                       'status': 'submitted' if result.returncode == 0 else 'failed',
                       'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, sort_keys=True)
        if result.returncode != 0:
            print(f'[preemption] Replacement submission failed (exit {result.returncode}).', flush=True)
            # Release the claim so a later attempt -- or another rep -- can retry.
            try:
                os.remove(marker_path)
            except FileNotFoundError:
                pass
            return False
        return True

    # ---------------------------------------------------------------- checkpoint save/load
    def _checkpoint_extra_state(self, cw_config):
        return {
            'checkpoint_version': 1,
            'num_iterations': self.n_completed,
            'interaction_step': self.interaction_step,
            'update_step': self.update_step,
            'update_counter': self.update_counter,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'wandb_run_id': self.wandb_run.id if self.wandb_run is not None else None,
            'config_fingerprint': _checkpoint_config_fingerprint(cw_config),
            'runtime': {
                'saved_at_unix': time.time(),
                'hostname': socket.gethostname(),
                'pid': os.getpid(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            },
        }

    @staticmethod
    def _link_or_copy(src_path, dst_path):
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return
        if not os.path.isfile(src_path):
            return
        tmp_dst = f'{dst_path}.tmp.{os.getpid()}'
        try:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
            try:
                os.link(src_path, tmp_dst)
            except OSError:
                shutil.copy2(src_path, tmp_dst)
            os.replace(tmp_dst, dst_path)
        finally:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)

    def _prune_checkpoints(self, directory, keep_basename):
        """Drop every checkpoint (and sidecar) in `directory` except `keep_basename`."""
        keep = {keep_basename, keep_basename + META_SUFFIX}
        for entry in os.listdir(directory):
            if entry in keep or not self._is_checkpoint_entry(entry):
                continue
            os.remove(os.path.join(directory, entry))

    def _save_checkpoint(self, cw_config, force=False):
        if self.save_model_dir is None:
            return None
        n = self.n_completed
        should_save = (
            force
            or n % self.save_model_interval == 0
            or n >= cw_config['iterations']
        )
        if not should_save:
            return None

        os.makedirs(self.save_model_dir, exist_ok=True)
        file_epoch = None if self.overwrite_checkpoints else n
        name = self.latest_checkpoint_name if file_epoch is None else f'{self.latest_checkpoint_name}_{file_epoch}'
        checkpoint_path = os.path.join(self.save_model_dir, name)

        save_checkpoint(
            checkpoint_path,
            self.agent,
            buffer=self.buffer,
            extra=self._checkpoint_extra_state(cw_config),
        )
        if self.overwrite_checkpoints:
            self._prune_checkpoints(self.save_model_dir, name)

        if self.resume_model_dir is not None:
            os.makedirs(self.resume_model_dir, exist_ok=True)
            # The sidecar goes second here too, for the same reason it does in
            # save_checkpoint: it certifies the checkpoint beside it is complete.
            self._link_or_copy(checkpoint_path, os.path.join(self.resume_model_dir, name))
            self._link_or_copy(
                checkpoint_path + META_SUFFIX,
                os.path.join(self.resume_model_dir, name + META_SUFFIX),
            )
            if self.overwrite_checkpoints:
                self._prune_checkpoints(self.resume_model_dir, name)

        print(
            f'[checkpoint] Saved: {checkpoint_path} '
            f'(n={n}, interaction_step={self.interaction_step}, update_step={self.update_step})',
            flush=True,
        )
        return checkpoint_path

    # ---------------------------------------------------------------- experiment lifecycle
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Before create_envs(), which is where a MuJoCo GL context first appears.
        _align_egl_device_with_cuda()

        self._skip_next_save_state = False
        self._active_run_lock_fd = None
        self._preemption_requested = False
        self._preemption_signal = None
        # NOT `self.run`: AbstractIterativeExperiment.run() is the driver that
        # calls iterate(), so binding a wandb Run to `self.run` shadows it and
        # cw2's `self.exp.run(c, r, logger)` fails with "'Run' object is not
        # callable". (Learned the hard way in the RLAC port.)
        self.wandb_run = None
        self.finished = False

        self.cfg = _compose_cfg(cw_config)
        cfg = self.cfg

        random.seed(cw_config['seed'])
        np.random.seed(cw_config['seed'])

        self.resume_model_dir = cw_config.get('resume_model_dir', None)
        if self.resume_model_dir is not None:
            self.resume_model_dir = os.path.abspath(self.resume_model_dir)

        auto_resume_enabled = _bool(cw_config.get('auto_resume_from_latest_checkpoint', False))
        if auto_resume_enabled and self._active_run_lock_enabled(cw_config):
            if not self._acquire_active_run_lock(cw_config):
                self._publish_checkpoint_ready(cw_config, None, status='skipped')
                raise cw_error.ExperimentSurrender({'active_run_lock_skipped': True})

        # --- checkpoint discovery -----------------------------------------
        resume_checkpoint_path = None
        if auto_resume_enabled:
            auto_resume_dir, latest_epoch, completed_dir, completed_epoch = self._find_latest_checkpoint(
                cw_config, self.resume_model_dir,
            )
            if completed_epoch is not None:
                print(
                    f'[checkpoint] Matching checkpoint is already complete at epoch '
                    f'{completed_epoch}/{cw_config["iterations"]}: {completed_dir}; skipping this run.',
                    flush=True,
                )
                self._publish_checkpoint_ready(cw_config, None, status='skipped')
                raise cw_error.ExperimentSurrender({'completed_checkpoint_skipped': True})
            elif latest_epoch is not None:
                resume_checkpoint_path = os.path.join(auto_resume_dir, self.latest_checkpoint_name)
                if not os.path.isfile(resume_checkpoint_path):
                    resume_checkpoint_path = os.path.join(
                        auto_resume_dir, f'{self.latest_checkpoint_name}_{latest_epoch}')
                print(f'[checkpoint] Auto-resuming from {resume_checkpoint_path} at n={latest_epoch}.', flush=True)
            else:
                print('[checkpoint] No checkpoint found; starting from scratch.', flush=True)

        self._register_preemption_handlers()

        # --- checkpointing config -------------------------------------------
        if cw_config.get('save_model_dir', None) is not None:
            self.save_model_dir = os.path.abspath(cw_config['save_model_dir'])
        else:
            self.save_model_dir = os.path.join(cw_config['_rep_log_path'], 'model')
        os.makedirs(self.save_model_dir, exist_ok=True)
        self.save_model_interval = max(cw_config['iterations'] // cw_config.get('num_checkpoints', 20), 1)
        self.overwrite_checkpoints = _bool(cw_config.get('overwrite_checkpoints', False))

        # --- envs / buffer / agent -------------------------------------------
        self.train_env, self.eval_env = create_envs(**cfg.env)

        self.buffer = create_buffer(
            observation_space=self.train_env.observation_space,
            action_space=self.train_env.action_space,
            **cfg.buffer,
        )
        self.buffer.reset()

        self.agent = create_agent(
            observation_space=self.train_env.observation_space,
            action_space=self.train_env.action_space,
            cfg=cfg.agent,
        )

        self.num_interaction_steps = int(cfg.num_interaction_steps)
        self.chunk_size = int(cw_config['params'].get('chunk_size', 5000))
        expected_iterations = math.ceil(self.num_interaction_steps / self.chunk_size)
        if int(cw_config['iterations']) != expected_iterations:
            print(
                f'[config] WARNING: iterations={cw_config["iterations"]} but '
                f'ceil(num_interaction_steps / chunk_size) = '
                f'ceil({self.num_interaction_steps} / {self.chunk_size}) = {expected_iterations}. '
                'Too few iterations truncate training silently; too many waste scheduler time.',
                flush=True,
            )

        # --- resumable bookkeeping (overwritten below if resuming) ------------
        self.n_completed = 0
        self.interaction_step = 0
        self.update_step = 0
        self.update_counter = 0
        self.last_update_info = {}
        self.observations = None
        self.timestep = None
        wandb_run_id = None
        wandb_resume = None

        if resume_checkpoint_path is not None:
            extra = load_checkpoint(resume_checkpoint_path, self.agent, buffer=self.buffer)
            self.n_completed = int(extra['num_iterations'])
            self.interaction_step = int(extra['interaction_step'])
            self.update_step = int(extra['update_step'])
            self.update_counter = float(extra.get('update_counter', 0))
            random.setstate(extra['python_random_state'])
            np.random.set_state(extra['numpy_random_state'])
            wandb_run_id = extra.get('wandb_run_id')
            wandb_resume = 'allow'

            # The simulator state of the episode that was in flight cannot be
            # restored, so the rollout restarts from a fresh episode. `timestep`
            # has to be a *post-reset* timestep rather than None: the training
            # loop only draws random actions while the buffer cannot be sampled
            # from, and after a resume it can, so `agent.sample_actions` runs from
            # the very first step. done=1 also makes RewardNormalizer zero its
            # running discounted return, which is correct at an episode boundary.
            self.observations, _ = self.train_env.reset()
            num_envs = int(cfg.num_train_envs)
            self.timestep = {
                'observation': self.observations,
                'action': np.zeros(self.train_env.action_space.shape, dtype=np.float32),
                'reward': np.zeros(num_envs, dtype=np.float32),
                'terminated': np.zeros(num_envs, dtype=np.float32),
                'truncated': np.ones(num_envs, dtype=np.float32),
                'next_observation': self.observations,
            }
        else:
            self.observations, _ = self.train_env.reset()

        # --- logging ----------------------------------------------------------
        exp_name = f"sd{cw_config['seed']:03d}"
        if os.environ.get('SLURM_JOB_ID'):
            exp_name += f"s_{os.environ['SLURM_JOB_ID']}"
        wandb_cfg = cw_config.get('wandb', {}) if isinstance(cw_config.get('wandb'), dict) else {}

        # Write the run's identity back into cfg before it is logged, so the
        # config attached to the W&B run says where the run actually went.
        if wandb_cfg.get('project'):
            cfg.project_name = wandb_cfg['project']
        cfg.group_name = wandb_cfg.get('group') or cw_config.get('_experiment_name') or cfg.group_name
        # `entity` from the cw2 config is authoritative *including* when it is
        # null: configs/online_rl.yaml ships the upstream authors' entity
        # ('draftrec'), and falling back to it would make wandb.init try to write
        # into somebody else's team. null means "this account's default entity".
        cfg.entity_name = wandb_cfg.get('entity')
        cfg.exp_name = exp_name
        cfg.save_path = self.save_model_dir

        self.logger = WandbTrainerLogger(
            cfg,
            name=exp_name,
            run_id=wandb_run_id,
            resume=wandb_resume,
        )
        self.wandb_run = self.logger.wandb_run

        # Initial evaluation, only on a fresh run: after a resume the run's wandb
        # step is already far past 0 and logging there again would be rejected as
        # non-monotonic.
        if resume_checkpoint_path is None:
            eval_info = evaluate(self.agent, self.eval_env, cfg.num_eval_episodes)
            self.logger.update_metric(**eval_info)
            self.logger.log_metric(step=0)
            self.logger.reset()

        self.progress_bar = tqdm(
            total=cw_config['iterations'], initial=self.n_completed, smoothing=0.1)

    # ---------------------------------------------------------------- training loop
    def _interaction_step(self):
        """One interaction step -- the body of run_online.py's training loop."""
        cfg = self.cfg
        self.interaction_step += 1
        i = self.interaction_step

        # While using random actions until buffer.can_sample(), we still feed data
        # into the agent so the normalization wrappers accumulate statistics.
        # `actions is None` only on the very first step of a fresh run, where
        # there is no previous timestep to act on yet (run_online.py leaves
        # `actions` unbound there, which is a NameError as soon as a buffer with
        # min_length=0 reports can_sample() on step one).
        actions = None
        if self.timestep:
            actions = self.agent.sample_actions(i, prev_timestep=self.timestep, training=True)
        if actions is None or self.buffer.can_sample() is False:
            actions = self.train_env.action_space.sample()

        next_observations, rewards, terminateds, truncateds, env_infos = self.train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(int(cfg.num_train_envs)):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos['final_obs'][env_idx]

        self.timestep = {
            'observation': self.observations,
            'action': actions,
            'reward': rewards,
            'terminated': terminateds,
            'truncated': truncateds,
            'next_observation': next_buffer_observations,
        }
        self.buffer.add(self.timestep)
        self.timestep['next_observation'] = next_observations
        self.observations = next_observations

        if not self.buffer.can_sample():
            return

        # update network; updates_per_interaction_step can be below 1.0
        self.update_counter += float(cfg.updates_per_interaction_step)
        while self.update_counter >= 1:
            batch = self.buffer.sample()
            self.last_update_info = self.agent.update(self.update_step, batch)
            self.logger.update_metric(**self.last_update_info)
            self.update_counter -= 1
            self.update_step += 1

        if i % int(cfg.evaluation_per_interaction_step) == 0:
            eval_info = evaluate(self.agent, self.eval_env, int(cfg.num_eval_episodes))
            self.logger.update_metric(**eval_info)

        if i % int(cfg.metrics_per_interaction_step) == 0:
            batch = self.buffer.sample()
            metrics_info = self.agent.get_metrics(batch, self.last_update_info)
            if metrics_info:
                self.logger.update_metric(**metrics_info)

        if int(cfg.num_record_episodes) > 0 and i % int(cfg.recording_per_interaction_step) == 0:
            video_info = record_video(self.agent, self.eval_env, int(cfg.num_record_episodes))
            self.logger.update_metric(**video_info)

        if i % int(cfg.logging_per_interaction_step) == 0:
            # Using env steps simplifies comparison with the numbers in the paper.
            # It is also monotonic across a requeue, which wandb requires: `i` is
            # restored from the checkpoint rather than restarting at 0.
            env_step = i * int(cfg.action_repeat) * int(cfg.num_train_envs)
            self.logger.log_metric(step=env_step)
            self.logger.reset()

    def _finish_run(self, cw_config):
        self.finished = True
        self.train_env.close()
        self.eval_env.close()
        if self.wandb_run is not None and self.wandb_run.url:
            with open(os.path.join(cw_config['_rep_log_path'], 'token.tk'), 'w') as f:
                f.write(self.wandb_run.url)
        print(
            f'[train] Finished: {self.interaction_step} interaction steps, '
            f'{self.update_step} gradient updates.', flush=True,
        )

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        target = min(self.interaction_step + self.chunk_size, self.num_interaction_steps)
        while self.interaction_step < target:
            self._interaction_step()

        if self.interaction_step >= self.num_interaction_steps and not self.finished:
            self._finish_run(cw_config)

        # `n` is cw2's per-process loop index: AbstractIterativeExperiment.run()
        # always iterates range(cw_config["iterations"]) from 0, so after a requeue
        # it restarts at 0. Assigning `n + 1` here would throw away the value
        # restored from the checkpoint; count cumulatively instead.
        self.n_completed += 1
        self.progress_bar.update(1)

        if self._preemption_requested:
            self._handle_preemption(cw_config)

        return {}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self._skip_next_save_state:
            self._skip_next_save_state = False
            return
        self._save_checkpoint(cw_config)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # A no-op when cancel-mode preemption already released it.
        self._release_active_run_lock()


if __name__ == '__main__':
    # EGL_DEVICE_ID is set in initialize() instead of here: with the
    # GPU-distributing scheduler this process is the parent of several workers
    # that each get a different GPU, so there is no single right answer yet.
    cw = cluster_work.ClusterWork(SimbaV2Experiment)
    prepare_rep_configs(cw.config)
    cw.run()
