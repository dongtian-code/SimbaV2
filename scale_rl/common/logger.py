from typing import Dict

from omegaconf import OmegaConf

import wandb


class WandbTrainerLogger(object):
    def __init__(
        self,
        cfg: Dict,
        name: str = None,
        run_id: str = None,
        resume: str = None,
        project: str = None,
        entity: str = None,
        group: str = None,
    ):
        """
        args:
            cfg: the resolved experiment config; logged as the run's config.
            name: explicit run name (default: wandb's generated one).
            run_id / resume: reattach to an existing run instead of creating a
                new one. A preempted-and-requeued job passes the id it stored in
                its checkpoint together with resume="allow", so the whole run
                stays a single continuous W&B curve across restarts rather than
                fragmenting into one run per SLURM allocation.
            project / entity / group: override the corresponding cfg fields;
                used by the cw2 launcher, which owns those names.
        """
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)

        self.wandb_run = wandb.init(
            project=project or cfg.project_name,
            entity=entity or cfg.entity_name,
            group=group or cfg.group_name,
            name=name,
            id=run_id,
            resume=resume,
            config=dict_cfg,
        )

        self.reset()

    def update_metric(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> Dict:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        self.wandb_run.log(log_data, step=step)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict = {}


class AverageMeterDict(object):
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
