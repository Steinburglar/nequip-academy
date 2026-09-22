"""NequIP datamodule that materializes teacher-labeled distillation data."""

import copy
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from hydra.utils import instantiate
from nequip.data.datamodule import ASEDataModule
from omegaconf import DictConfig, ListConfig, OmegaConf

from nequip_academy.data.paths import split_file
from nequip_academy.sample.split import DEFAULT_SPLIT

logger = logging.getLogger(__name__)


def _to_container(value: Any) -> Any:
    """Convert OmegaConf containers to plain containers, leaving objects alone."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _as_path_list(value: Any) -> list:
    """Normalize a single path or a list of paths to a list of strings."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in value]


def _generated_test_fraction(generation_config: Any) -> float:
    """How much of the generated dataset the configured split gives to ``test``.

    Read in ``__init__``, before the full validation in :meth:`_generation_config`, so
    it stays tolerant of a config that turns out to be malformed -- that error belongs
    to the later check, not this one.
    """
    if not isinstance(generation_config, dict):
        return 0.0
    split = generation_config.get("split") or DEFAULT_SPLIT
    if not isinstance(split, dict):
        return 0.0
    try:
        return float(split.get("test", 0.0))
    except (TypeError, ValueError):
        return 0.0


TEACHER_LABELED_TEST_WARNING = (
    "This run's test split is TEACHER-LABELED.\n"
    "  Its metrics measure how closely the student reproduces the teacher, NOT how "
    "accurate the student is.\n"
    "  Do not report them as the student's test error. For that, label a held-out set "
    "with your ground-truth method\n"
    "  and pass it as `data.test_file_path`, with the generated test fraction at 0."
)


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
    test_file_path
        Path (or list of paths) to an externally labeled held-out set, exactly as any
        other ``ASEDataModule`` takes it. This is the intended way to get a test set:
        generation makes train and val only, and the test set carries whatever labels
        the user considers ground truth. It need not live under ``dataset_path``, it is
        never written to, and it goes through the same ``transforms``, ``key_mapping``
        and ``include_keys`` as the generated files. Relative paths resolve against the
        launch directory, because hydra does not chdir.
    teacher_labeled_test
        Opt in to a generated, teacher-labeled test split. Off by default, and required
        whenever ``generation.split`` gives ``test`` a nonzero share, because such a
        test set measures agreement with the teacher rather than accuracy. Logs a
        warning whenever it is on.
    """

    def __init__(
        self,
        seed: int,
        dataset_path: Union[str, Path],
        generation: Union[dict, DictConfig],
        teacher: Optional[Any] = None,
        state_interval: int = 1,
        test_file_path: Optional[Union[str, Path, list]] = None,
        teacher_labeled_test: bool = False,
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
        owned = [s for s in ("train", "val") if f"{s}_file_path" in kwargs]
        if owned:
            raise ValueError(
                "`DistillationDataModule` generates the training and validation sets, "
                f"so it owns their paths through `dataset_path`; do not set "
                f"{', '.join(f'`data.{s}_file_path`' for s in owned)}. "
                "`data.test_file_path` IS accepted -- that is how you supply an "
                "externally labeled test set."
            )

        self.dataset_path = Path(dataset_path)
        self.generation_config = _to_container(generation)
        self.teacher_config = _to_container(teacher)
        self.state_interval = int(state_interval)
        if self.state_interval < 1:
            raise ValueError(
                f"`state_interval` must be at least 1, got {state_interval!r}"
            )

        external_test = _as_path_list(test_file_path)
        generated_test_fraction = _generated_test_fraction(self.generation_config)
        self.teacher_labeled_test = bool(teacher_labeled_test)

        if external_test and generated_test_fraction > 0:
            raise ValueError(
                f"`data.test_file_path` supplies a test set, but the generation split "
                f"also allocates test={generated_test_fraction} to teacher-labeled "
                "data. Pick one: drop the test share from `generation.split` (giving "
                "it to train or val), or drop `data.test_file_path`."
            )
        if generated_test_fraction > 0 and not self.teacher_labeled_test:
            raise ValueError(
                f"`generation.split` allocates test={generated_test_fraction}, which "
                "would make the test set TEACHER-LABELED. Its metrics would measure "
                "agreement with the teacher, not accuracy, and reporting them as the "
                "student's test error is wrong. Either pass a ground-truth-labeled "
                "held-out set as `data.test_file_path` and zero the test share, or, if "
                "you really do want to measure distillation fidelity, set "
                "`data.teacher_labeled_test: true`."
            )
        if self.teacher_labeled_test and generated_test_fraction <= 0:
            raise ValueError(
                "`data.teacher_labeled_test` is set but `generation.split` allocates "
                "no test share, so there is no teacher-labeled test set to opt into."
            )
        missing = [path for path in external_test if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                f"`data.test_file_path` does not exist: {missing}. Paths resolve "
                "against the launch directory, because hydra does not chdir."
            )

        if external_test:
            test_paths = external_test
        elif generated_test_fraction > 0:
            test_paths = [str(split_file(self.dataset_path, "test"))]
            logger.warning(TEACHER_LABELED_TEST_WARNING)
        else:
            test_paths = []

        super().__init__(
            seed=seed,
            train_file_path=[str(split_file(self.dataset_path, "train"))],
            val_file_path=[str(split_file(self.dataset_path, "val"))],
            test_file_path=test_paths,
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
        part a standalone generation script would keep (the portability invariant
        in CLAUDE.md).
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
        if self.teacher_labeled_test:
            logger.warning(TEACHER_LABELED_TEST_WARNING)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
