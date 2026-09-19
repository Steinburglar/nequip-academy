"""NequIP datamodule that materializes teacher-labeled distillation data."""

import copy
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from hydra.utils import instantiate
from nequip.data.datamodule import ASEDataModule
from omegaconf import DictConfig, ListConfig, OmegaConf

from nequip_extension_template.data.paths import SPLITS, split_file

logger = logging.getLogger(__name__)


def _to_container(value: Any) -> Any:
    """Convert OmegaConf containers to plain containers, leaving objects alone."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


class DistillationDataModule(ASEDataModule):
    """Generate a distillation dataset, then expose it as a normal ASE datamodule.

    This is the first, intentionally thin, datamodule boundary: generation still uses
    the existing sampler classes and state file. The datamodule owns the NequIP data
    interface and wires the generated ``train.extxyz``/``val.extxyz``/``test.extxyz``
    files into :class:`nequip.data.datamodule.ASEDataModule`.

    Parameters
    ----------
    sample_path
        Directory holding generated split files and sampler state.
    generation
        Hydra config for the current sampler class, for example
        ``nequip_extension_template.sample.RattleSampler``. It should contain the
        sampling parameters such as ``base_frames`` and split settings, but not
        ``sample_path``.
    teacher
        Hydra config for the teacher ASE calculator. The datamodule injects this as
        the sampler's ``calculator``. For compatibility, ``generation.calculator`` is
        also accepted when ``teacher`` is omitted.
    state_interval
        Forwarded to the sampler as its checkpoint cadence.
    """

    def __init__(
        self,
        seed: int,
        sample_path: Union[str, Path],
        generation: Union[dict, DictConfig],
        teacher: Optional[Any] = None,
        state_interval: int = 1,
        transforms: list = [],
        ase_args: dict = {},
        include_keys: Optional[list] = [],
        exclude_keys: Optional[list] = [],
        key_mapping: Optional[dict] = {},
        train_dataloader: dict = {},
        val_dataloader: dict = {},
        test_dataloader: dict = {},
        predict_dataloader: dict = {},
        stats_manager: Optional[dict] = None,
        **kwargs,
    ):
        if "split_dataset" in kwargs and kwargs["split_dataset"]:
            raise ValueError(
                "`DistillationDataModule` writes pre-split train/val/test files; "
                "do not also set `data.split_dataset`."
            )
        if any(f"{split}_file_path" in kwargs for split in SPLITS):
            raise ValueError(
                "`DistillationDataModule` owns train/val/test file paths through "
                "`sample_path`; do not set `data.train_file_path`, "
                "`data.val_file_path`, or `data.test_file_path`."
            )

        self.sample_path = Path(sample_path)
        self.generation_config = _to_container(generation)
        self.teacher_config = _to_container(teacher)
        self.state_interval = int(state_interval)
        if self.state_interval < 1:
            raise ValueError(
                f"`state_interval` must be at least 1, got {state_interval!r}"
            )

        split_paths = {
            split: str(split_file(self.sample_path, split)) for split in SPLITS
        }

        super().__init__(
            seed=seed,
            train_file_path=[split_paths["train"]],
            val_file_path=[split_paths["val"]],
            test_file_path=[split_paths["test"]],
            predict_file_path=[],
            split_dataset=[],
            transforms=transforms,
            ase_args=ase_args,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            key_mapping=key_mapping,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            predict_dataloader=predict_dataloader,
            stats_manager=stats_manager,
        )

    def _sampler_config(self) -> dict:
        """Return the sampler config used for instantiation and provenance."""
        if not isinstance(self.generation_config, dict):
            raise TypeError(
                "`generation` must be a Hydra config dictionary with a `_target_`"
            )
        sampler_config = copy.deepcopy(self.generation_config)
        if "_target_" not in sampler_config:
            raise ValueError("`data.generation._target_` must be provided")

        has_calculator = "calculator" in sampler_config
        if self.teacher_config is not None and has_calculator:
            raise ValueError(
                "provide the teacher calculator either as `data.teacher` or as "
                "`data.generation.calculator`, not both"
            )
        if self.teacher_config is not None:
            sampler_config["calculator"] = copy.deepcopy(self.teacher_config)
        if "calculator" not in sampler_config:
            raise ValueError(
                "`DistillationDataModule` needs a teacher calculator in "
                "`data.teacher` or `data.generation.calculator`"
            )
        return sampler_config

    def _instantiate_sampler(self, sampler_config: dict):
        """Build the configured sampler, WITHOUT its teacher.

        The teacher is left out so that `generate()` can decide whether it is needed
        at all. `sampler_config` still carries the teacher's own config, because that
        is provenance and has to be compared whether or not the model gets loaded.
        """
        config = {k: v for k, v in sampler_config.items() if k != "calculator"}
        sampler = instantiate(
            config,
            sample_path=str(self.sample_path),
            state_interval=self.state_interval,
        )
        sampler.sampler_config = sampler_config
        return sampler

    def prepare_data(self) -> None:
        """Generate or resume the sampled dataset before NequIP loads it.

        Two halves, deliberately. The first is hydra plumbing, specific to running
        under `nequip-train`. The second names only local variables, so it is the
        part a standalone generation script would keep (`planning.md` 11.1a).
        """
        # --- adapter: hydra config -> objects. A script replaces this half with
        #     ordinary construction, so everything touching `self` belongs here.
        sampler_config = self._sampler_config()
        teacher_config = copy.deepcopy(sampler_config["calculator"])
        sampler = self._instantiate_sampler(sampler_config)
        teacher_factory = lambda: instantiate(teacher_config)  # noqa: E731
        logger.info(f"preparing distillation dataset -> {self.sample_path}")

        # --- pipeline: names only local variables, so it is paste-able verbatim
        n_total = sampler.generate(teacher_factory=teacher_factory)
        counts = ", ".join(f"{s}={n}" for s, n in sampler.split_counts.items())
        logger.info(
            f"{sampler.sample_path} holds {n_total} labeled structures ({counts}); "
            f"{n_total - sampler.n_resumed} from this run, "
            f"{sampler.n_resumed} already present"
        )
        sampler.release_calculator()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
