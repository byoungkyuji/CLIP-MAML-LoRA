import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from trainers.lora import LoRA
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.lora
from dassl.evaluation import build_evaluator
from ray import tune
from ray.air import Checkpoint, session,CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.pb2 import PB2
from functools import partial
def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.LoRA = CN()
    cfg.TRAINER.LoRA.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LoRA.POSITION="all"
    cfg.TRAINER.LoRA.ENCODER="both"
    cfg.TRAINER.LoRA.PARAMS=["q","k","v"]
    cfg.TRAINER.LoRA.RANK=2
    cfg.TRAINER.LoRA.ALPHA=1
    cfg.TRAINER.LoRA.DROPOUT_RATE=0.25
    cfg.TRAINER.LoRA.CTX_INIT=['a photo of a {}.']
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TEST.FINAL_MODEL = "best_val"

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def train_hpo(trainer,cfg,config):
    trainer.build_model(config)
    trainer.evaluator = build_evaluator(cfg, lab2cname=trainer.lab2cname)
    trainer.train()

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    #setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        #trainer.train()
        
        config = {
        "r":tune.choice([2,16,32]),
        "dropout_rate":tune.choice([0.25,0.5]),
        "params":tune.choice([["q"],["k"],["v"],["o"],["q","v"],["q","k","v"],["q","v","k","o"]])
        }

        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric="accuracy",
            mode="max",
            max_t=cfg.OPTIM.MAX_EPOCH,
            grace_period=2,
            reduction_factor=4,
        )
        # scheduler = PB2(
        # time_attr='training_iteration',
        # metric="accuracy",
        # mode="max",
        # perturbation_interval=5,
        # hyperparam_bounds={"lr": [1e-4, 1e-2 ]},
        # )
        result = tune.run(
            partial(train_hpo,trainer, cfg),
            resources_per_trial={"cpu": 8, "gpu": 0.5},
            config=config,
            num_samples=16,
            scheduler=scheduler,
            checkpoint_config=CheckpointConfig(num_to_keep = 1, checkpoint_score_attribute="accuracy",checkpoint_score_order ="max")
        )
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
