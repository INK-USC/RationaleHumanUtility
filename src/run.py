import os, shutil
from typing import Tuple, Optional

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import open_dict, DictConfig
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.data import dataset_info, monitor_dict
from src.utils.logging import get_logger, get_neptune_logger
from src.utils.callbacks import BestPerformance

from src.model.gpt3 import GPT3Runner


def get_callbacks(cfg: DictConfig):

    monitor = monitor_dict[cfg.data.dataset]
    mode = cfg.data.mode
    callbacks = []

    if cfg.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                min_delta=0.00,
                patience=cfg.training.patience,
                verbose=False,
                mode=mode
            )
        )

    callbacks.append(BestPerformance(monitor=monitor, mode=mode))

    if cfg.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                dirpath=os.path.join(cfg.save_dir, 'checkpoints'),
                save_top_k=1,
                mode=mode,
                verbose=True,
                save_last=False,
                save_weights_only=True,
            )
        )



    return callbacks


logger = get_logger(__name__)


def build(cfg) -> Tuple[pl.LightningDataModule, pl.LightningModule, pl.Trainer]:
    dm = instantiate(
        cfg.data,
        arch=cfg.model.arch,
        save_dir=cfg.save_dir,
    )
    dm.setup(stage=None, splits=cfg.training.eval_splits.split(","))

    logger.info(f'load {cfg.data.dataset} <{cfg.data._target_}>')

    run_logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    with open_dict(cfg):
        if cfg.debug or cfg.logger.offline:
            exp_dir = cfg.logger.name
            # cfg.logger.neptune_exp_id = cfg.logger.name
        else:
            if cfg.logger.logger == "neptune":
                print("HERE------")
                print(run_logger)
                # print(run_logger.version)
                print(run_logger.save_dir)
                print(run_logger.name)
                # exp_dir = run_logger.name
                print(dir(run_logger))
                # print(run_logger.experiment)
                print(run_logger._run_name)
                print(run_logger.experiment["sys/id"].fetch())

                exp_dir = run_logger.experiment["sys/id"].fetch()
                cfg.logger.neptune_exp_id = run_logger.experiment["sys/id"].fetch()
            else:
                raise NotImplementedError

        print(cfg.save_dir, exp_dir)
        cfg.save_dir = os.path.join(cfg.save_dir, exp_dir)
        os.makedirs(cfg.save_dir, exist_ok=True)

        if exp_dir != "test":
            # copy hydra configs
            shutil.copytree(
                os.path.join(os.getcwd(), ".hydra"),
                os.path.join(cfg.save_dir, "hydra")
            )

    logger.info(f"saving to {cfg.save_dir}")

    model = instantiate(
        cfg.model,
        # num_classes=dataset_info[cfg.data.dataset]['num_classes'],
        load_exp_id=cfg.data.load_exp_id,
        save_exp_id=cfg.logger.neptune_exp_id,
        _recursive_=False
    )

    logger.info(f'load {cfg.model.arch} <{cfg.model._target_}>')

    cb = get_callbacks(cfg)

    trainer = instantiate(
        cfg.trainer,
        callbacks=cb,
        logger=run_logger,
        _convert_="all",
    )

    return dm, model, trainer


def build_gpt3(cfg) -> pl.LightningDataModule:
    dm = instantiate(
        cfg.data,
        arch=cfg.model.arch,
        save_dir=cfg.save_dir,
    )
    dm.setup(stage=None, splits=cfg.training.eval_splits.split(","))

    logger.info(f'load {cfg.data.dataset} <{cfg.data._target_}>')
    
    run_logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    with open_dict(cfg):
        if cfg.debug or cfg.logger.offline:
            exp_dir = cfg.logger.name
            # cfg.logger.neptune_exp_id = cfg.logger.name
        else:
            if cfg.logger.logger == "neptune":
                print("HERE------")
                print(run_logger)
                # print(run_logger.version)
                print(run_logger.save_dir)
                print(run_logger.name)
                # exp_dir = run_logger.name
                print(dir(run_logger))
                # print(run_logger.experiment)
                print(run_logger._run_name)
                print(run_logger.experiment["sys/id"].fetch())

                exp_dir = run_logger.experiment["sys/id"].fetch()
                cfg.logger.neptune_exp_id = run_logger.experiment["sys/id"].fetch()
            else:
                raise NotImplementedError

        print(cfg.save_dir, exp_dir)
        cfg.save_dir = os.path.join(cfg.save_dir, exp_dir)
        os.makedirs(cfg.save_dir, exist_ok=True)

        if exp_dir != "test":
            # copy hydra configs
            shutil.copytree(
                os.path.join(os.getcwd(), ".hydra"),
                os.path.join(cfg.save_dir, "hydra")
            )

    logger.info(f"saving to {cfg.save_dir}")

    model = GPT3Runner(
        dataset = dm.test_dataloader().dataset,
        arch=cfg.model.arch,
        gen_mode=cfg.data.gen_mode,
        incontext=cfg.data.incontext,
        prompt_type=cfg.data.prompt_type,
        save_dir=cfg.save_dir,
        dataname=cfg.data.dataset,
        )

    trainer = instantiate(
        cfg.trainer,
        callbacks=get_callbacks(cfg),
        # checkpoint_callback=cfg.save_checkpoint,
        logger=run_logger,
        _convert_="all",
    )

    return model, trainer, dm.test_dataloader()


def restore_config_params(model, cfg: DictConfig):
    for key, val in cfg.model.items():
        setattr(model, key, val)

    for key, val in cfg.data.items():
        if key in ['load_exp_id']:
            setattr(model, key, val)

    if cfg.model.save_outputs:
        assert cfg.model.save_exp_id in cfg.training.ckpt_path
        setattr(model, key, val)

    logger.info('Restored params from model config.')

    return model


def run(cfg: DictConfig) -> Optional[float]:
    pl.seed_everything(cfg.seed)

    if cfg.model.model_type == "gpt3":
        model, trainer, loader = build_gpt3(cfg)
        trainer.test(model=model, dataloaders=loader)
        # gpt3_runner(cfg, dm.test_dataloader(), logger)
        
    else:
        dm, model, trainer = build(cfg)
        pl.seed_everything(cfg.seed)

        if cfg.save_rand_checkpoint:
            ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', 'rand.ckpt')
            logger.info(f"Saving randomly initialized model to {ckpt_path}")
            trainer.model = model
            trainer.save_checkpoint(ckpt_path)
        elif not cfg.training.evaluate_ckpt:
            # either train from scratch, or resume training from ckpt
            if cfg.training.finetune_ckpt:
                assert cfg.training.ckpt_path
                save_dir = '/'.join(cfg.save_dir.split('/')[:-1])
                ckpt_path = os.path.join(save_dir, cfg.training.ckpt_path)
                model = model.load_from_checkpoint(ckpt_path, strict=False)
                model = restore_config_params(model, cfg)
                logger.info(f"Loaded checkpoint (for fine-tuning) from {ckpt_path}")

            trainer.fit(model=model, datamodule=dm)
            if getattr(cfg, "tune_metric", None):
                metric = trainer.callback_metrics[cfg.tune_metric].detach()
                logger.info(f"best metric {metric}")
                return metric
        else:
            # evaluate the pretrained model on the provided splits
            assert cfg.training.ckpt_path
            num_levels_remove = 1
            save_dir = '/'.join(cfg.save_dir.split('/')[:-num_levels_remove])
            ckpt_path = os.path.join(save_dir, cfg.training.ckpt_path)
            model = model.load_from_checkpoint(ckpt_path, strict=False)
            logger.info(f"Loaded checkpoint for evaluation from {cfg.training.ckpt_path}")
            model = restore_config_params(model, cfg)
            print('Evaluating loaded model checkpoint...')
            for split in cfg.training.eval_splits.split(','):
                print(f'Evaluating on split: {split}')
                if split == 'train':
                    loader = dm.train_dataloader()
                elif split == 'dev':
                    loader = dm.val_dataloader(test=True)
                elif split == 'test':
                    loader = dm.test_dataloader()

                trainer.test(model=model, dataloaders=loader)


        """
srun --gres=gpu:2080:1 -t 1-10 python main.py \
data=qed \
data.gen_mode=I-OR \
data.incontext=feb_6 \
data.prompt_type=cot \
model=lm \
model.model_type=gpt3 \
model.arch=davinci-instruct-beta \
training.eval_splits=test \
setup.eval_batch_size=1 \
setup.num_workers=3 \
trainer.max_steps=200 \
seed=0 


CUDA_VISIBLE_DEVICES=6 python main.py \
data=strategyqa \
data.gen_mode=I-OR \
data.incontext=feb \
data.prompt_type=feb \
model=lm \
model.model_type=gpt3 \
model.arch=davinci-instruct-beta \
training.eval_splits=test \
setup.eval_batch_size=1 \
setup.num_workers=3 \
seed=0 \
data.presample=200
"""

"""
Ziyi's command
HYDRA_FULL_ERROR=1 srun --gres=gpu:8000:2 -t 1-10 python main.py \
data=strategyqa \
data.gen_mode=I-OR \
data.incontext=None \
data.prompt_type=squadt5 \
data.num_train=48 \
data.num_train_seed=0 \
model=lm \
model.arch=t5-3b \
setup.train_batch_size=4 \
setup.eval_batch_size=4 \
setup.num_workers=3 \
setup.precision=16 \
trainer.accelerator=gpu \
trainer.devices=2 \
trainer.strategy=deepspeed_stage_3 \
logger.offline=True \
seed=0



CUDA_VISIBLE_DEVICES=0 \
python main.py \
data=openbookqa \
data.gen_mode=I-OR \
data.incontext=None \
data.prompt_type=infilling \
model=lm \
model.arch=t5-large \
setup.train_batch_size=4 \
setup.eval_batch_size=4 \
setup.num_workers=3 \
seed=0 \
trainer.max_steps=20000000 \
trainer.accelerator=gpu \
trainer.devices=1 \
setup.eff_train_batch_size=4 \
setup.accumulate_grad_batches=1 \
trainer.check_val_every_n_epoch=25 \
trainer.max_epochs=25 \
training.patience=25
"""