# hydra_main.py
from __future__ import annotations
import os, sys, subprocess, importlib, inspect
from pathlib import Path
from typing import Callable, Optional
from omegaconf import DictConfig
import hydra

def _find_entry(mod) -> Optional[Callable]:
    # Try common names used by training scripts
    for name in ("train", "main", "run", "app"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None

def _call_entry(fn: Callable, cfg: DictConfig):
    # Try to pass cfg if the function accepts an argument, otherwise call with no args
    try:
        if fn.__code__.co_argcount >= 1:
            return fn(cfg)  # many projects accept a config dict/object
    except Exception:
        pass
    return fn()

@hydra.main(version_base="1.3", config_path="conf", config_name="online")
def main(cfg: DictConfig):
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

    # Pick which runner to use (online/offline/parallel), default "online"
    runner = cfg.get("runner", "online")
    script = dict(
        online="run_online",
        offline="run_offline",
        parallel="run_parallel",
    ).get(runner, "run_online")

    # 1) Try to import and call an entry function directly
    try:
        mod = importlib.import_module(script)
        fn = _find_entry(mod)
        if fn:
            return _call_entry(fn, cfg)
    except Exception:
        # If import or direct call fails, fall back to subprocess
        pass

    # 2) Fallback: execute the original script as a new process
    root = Path(hydra.utils.get_original_cwd())
    cmd = [sys.executable, "-u", str(root / f"{script}.py")]
    # If your original script accepts CLI args, append them here based on cfg.
    # e.g., cmd += [f"--seed={cfg.seed}", f"--task={cfg.task}"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
