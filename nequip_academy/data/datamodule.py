"""NequIP datamodule that materializes teacher-labeled distillation data."""

import copy
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from hydra.utils import instantiate
from nequip.data.datamodule import ASEDataModule
from omegaconf import DictConfig, ListConfig, OmegaConf

from nequip_academy.data.paths import SPLITS, split_file

logger = logging.getLogger(__name__)


def _to_container(value: Any) -> Any:
    """Convert OmegaConf containers to plain containers, leaving objects alone."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


class DistillationDataModule(ASEDataModule):
    """Generate a distillation dataset, then expose it as a normal ASE datamodule.

    This is the first, intentionally thin, datamodule boundary: generation still uses
    the existing generator classes and state file. The datamodule owns the NequIP data
    interface and wires the generated ``train.extxyz``/``val.extxyz``/``test.extxyz``
    files into :class:`nequip.data.datamodule.ASEDataModule`.

    Parameters
    ----------
    dataset_path
        Directory holding generated split files and generator state.
    generation
        Hydra config for the current generator class, for example
        ``nequip_academy.sample.RattleGenerator``. It should contain the
        sampling parameters such as ``base_frames`` and split settings, but not
        ``dataset_path``.
    teacher
        Hydra config for the teacher ASE calculator. The datamodule injects this as
        the generator's ``calculator``. For compatibility, ``generation.calculator`` is
        also accepted when ``teacher`` is omitted.
    state_interval
        Forwarded to the generator as its checkpoint cadence.
    """

    def __init__(
        self,
        seed: int,
        dataset_path: Union[str, Path],
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
                "`dataset_path`; do not set `data.train_file_path`, "
                "`data.val_file_path`, or `data.test_file_path`."
            )

        self.dataset_path = Path(dataset_path)
        self.generation_config = _to_container(generation)
        self.teacher_config = _to_container(teacher)
        self.state_interval = int(state_interval)
        if self.state_interval < 1:
            raise ValueError(
                f"`state_interval` must be at least 1, got {state_interval!r}"
            )

        split_paths = {
            split: str(split_file(self.dataset_path, split)) for split in SPLITS
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

    def _generation_config(self) -> dict:
        """Return the generator config used for instantiation and provenance."""
        if not isinstance(self.generation_config, dict):
            raise TypeError(
                "`generation` must be a Hydra config dictionary with a `_target_`"
            )
        generation_config = copy.deepcopy(self.generation_config)
        if "_target_" not in generation_config:
            raise ValueError("`data.generation._target_` must be provided")

        has_calculator = "calculator" in generation_config
        if self.teacher_config is not None and has_calculator:
            raise ValueError(
                "provide the teacher calculator either as `data.teacher` or as "
                "`data.generation.calculator`, not both"
            )
        if self.teacher_config is not None:
            generation_config["calculator"] = copy.deepcopy(self.teacher_config)
        if "calculator" not in generation_config:
            raise ValueError(
                "`DistillationDataModule` needs a teacher calculator in "
                "`data.teacher` or `data.generation.calculator`"
            )
        return generation_config

    def _instantiate_generator(self, generation_config: dict):
        """Build the configured generator, WITHOUT its teacher.

        The teacher is left out so that `generate()` can decide whether it is needed
        at all. `generation_config` still carries the teacher's own config, because that
        is provenance and has to be compared whether or not the model gets loaded.
        """
        config = {k: v for k, v in generation_config.items() if k != "calculator"}
        generator = instantiate(
            config,
            dataset_path=str(self.dataset_path),
            state_interval=self.state_interval,
        )
        generator.generation_config = generation_config
        return generator

    def prepare_data(self) -> None:
        """Generate or resume the sampled dataset before NequIP loads it.

        Two halves, deliberately. The first is hydra plumbing, specific to running
        under `nequip-train`. The second names only local variables, so it is the
        part a standalone generation script would keep (`planning.md` 11.1a).
        """
        # --- adapter: hydra config -> objects. A script replaces this half with
        #     ordinary construction, so everything touching `self` belongs here.
        generation_config = self._generation_config()
        teacher_config = copy.deepcopy(generation_config["calculator"])
        generator = self._instantiate_generator(generation_config)
        teacher_factory = lambda: instantiate(teacher_config)  # noqa: E731
        logger.info(f"preparing distillation dataset -> {self.dataset_path}")

        # --- pipeline: names only local variables, so it is paste-able verbatim
        n_total = generator.generate(teacher_factory=teacher_factory)
        counts = ", ".join(f"{s}={n}" for s, n in generator.split_counts.items())
        logger.info(
            f"{generator.dataset_path} holds {n_total} labeled structures ({counts}); "
            f"{n_total - generator.n_resumed} from this run, "
            f"{generator.n_resumed} already present"
        )
        generator.release_calculator()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
