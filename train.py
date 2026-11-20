import argparse
import itertools
import json
import os
import signal
from dataclasses import asdict

import lightning.pytorch as pl
from lightning.pytorch import loggers
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities.seed import seed_everything

from config import Config
from data_module import DataModule, DPLightningDataModule
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to train.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default=None,
        help="Specify which optimizer to use.",
    )
    parser.add_argument(
        "--use-dp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable differential privacy training.",
    )
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=None,
        help="Target epsilon for differential privacy training. Ignored if --use-dp is not set.",
    )
    parser.add_argument(
        "--target-delta",
        type=float,
        default=None,
        help="Target delta for differential privacy training. Ignored if --use-dp is not set.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Maximum gradient norm for differential privacy training. Ignored if --use-dp is not set.",
    )
    parser.add_argument(
        "--dp-adam-bc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Adam bias correction for DP Adam optimizer. Ignored if --use-dp is not set and optimizer is not 'adam'.",
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Number of training samples to use.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Path to a sweep JSON file for hyperparameter runs.",
    )
    parser.add_argument(
        "--sweep-index",
        type=int,
        default=None,
        help="Index of the sweep combination to run.",
    )
    parser.add_argument(
        "--list-sweep",
        action="store_true",
        help="List all sweep combinations and exit.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root path to the CIFAR-10 dataset. Dataset will be downloaded if not present.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Number of processes used for the PyTorch dataloader.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Directory to save the training logs.",
    )
    parser.add_argument(
        "--hpc-tmp-dir",
        type=str,
        default="hpc_tmp/",
        help="Directory to use for temporary files on HPC.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="Run n batches of training and validation for debug.",
    )

    return parser.parse_args()


def apply_overrides(base, overrides):
    base_dict = asdict(base)
    allowed = set(base_dict)
    updates = {k: v for k, v in overrides.items() if v is not None and k in allowed}

    use_dp_value = updates.get("use_dp", base.use_dp)
    if not use_dp_value:
        for key in ("target_epsilon", "target_delta", "max_grad_norm", "dp_adam_bc"):
            updates.pop(key, None)

    return Config(**{**base_dict, **updates})


def train(config, args):
    name = f"lr_{config.learning_rate}_opt_{config.optimizer}"
    if config.use_dp:
        name += f"_eps_{config.target_epsilon}_mgn_{config.max_grad_norm}"
        if config.optimizer == "adam":
            name += f"_bc_{config.dp_adam_bc}"

    seed_everything(943264, workers=True)

    data = DataModule(
        dataset_dir=args.dataset_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        **asdict(config),
    )

    if config.use_dp:
        data = DPLightningDataModule(data)

    model = Model(**asdict(config))

    plugins = None
    if "SLURM_JOB_ID" in os.environ:
        plugins = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)]

    trainer = pl.Trainer(
        accelerator="cpu",
        default_root_dir=(
            os.path.join(args.hpc_tmp_dir, name) if args.hpc_tmp_dir else None
        ),
        logger=loggers.CSVLogger(f"{args.log_dir}/", name=name),
        precision="32-true",
        fast_dev_run=False if args.debug is None else args.debug,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_checkpointing=False,
        deterministic=True,
        plugins=plugins,
    )

    trainer.fit(model, datamodule=data, weights_only=False)


def main():
    args = parse_args()

    if args.sweep:
        base_config = Config()

        with open(args.sweep, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        keys = list(data.keys())
        values = [v if isinstance(v, list) else [v] for v in data.values()]
        combos = list(itertools.product(*values))
        configs = [
            apply_overrides(base_config, dict(zip(keys, combo))) for combo in combos
        ]

        if args.list_sweep:
            for idx, config in enumerate(configs):
                print(f"[{idx}] {asdict(config)}")
            return

        if args.sweep_index is not None:
            train(configs[args.sweep_index], args)
        else:
            for config in configs:
                train(config, args)

    else:
        overrides = {
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "optimizer": args.optimizer,
            "use_dp": args.use_dp,
            "target_epsilon": args.target_epsilon,
            "target_delta": args.target_delta,
            "max_grad_norm": args.max_grad_norm,
            "dp_adam_bc": args.dp_adam_bc,
            "train_subset_size": args.train_subset_size,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
        }

        base_config = apply_overrides(Config(), overrides)
        train(base_config, args)


if __name__ == "__main__":
    main()
