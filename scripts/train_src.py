import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from models.simplenet import SimpleNet, seed_everything
from models.loss import SupConLoss
import warnings
import ssl
from data.grain import GrainDataset, initialize_worker
import utils.utils as utils
import models.backbones as backbones


warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = F.pairwise_distance(anchor, positive, 2)
    d_an = F.pairwise_distance(anchor, negative, 2)
    loss = F.relu(d_ap - d_an + margin)
    return loss.mean()


transform = transforms.Compose([
                    transforms.Lambda(lambda t: (t + 1) / 2),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


@hydra.main(config_path="../configs", config_name="train_src", version_base=None)
def main(args):
    seed_everything(42)
    args = DictConfig(OmegaConf.to_container(args, resolve=True))
    args['models']['ica_path'] = args['ica_path']
    args['models']['ablation'] = args['ablation']
    if args['five_split'] is True:
        args['imagesize'] = args['resize'] = [224, 224]
        args['models']['input_shape'] = [3, 224, 224]
    else:
        args['imagesize'] = args['resize'] = [224, 1120]
        args['models']['input_shape'] = [3, 224, 1120]
    args['models']['five_split'] = args['five_split']
    if args.models.add_ica and not args.five_split:
        raise ValueError('The rotation STN requires five_split=true (224x224 views)')
    logger.info(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print(accelerator.device)
    # dataset
    train_dataset = GrainDataset(args.train_data_dir, **args)
    test_dataset = GrainDataset(args.train_data_dir, is_train=False, **args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=initialize_worker, persistent_workers=args.num_workers > 0)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)

    # model
    model_cfg = OmegaConf.to_container(args.models)
    model_cfg['input_shape'] = train_dataset.imagesize
    model_cfg['device'] = accelerator.device
    simplenet = SimpleNet(accelerator.device)
    backbone = backbones.load(args.backbone_name)
    backbone.name, backbone.seed = args.backbone_name, None
    simplenet.load(backbone=backbone, **model_cfg)

    simplenet, train_dataloader, test_dataloader = accelerator.prepare(
        simplenet, train_dataloader, test_dataloader
    )
    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
            args.batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Train_data_dir: {args.train_data_dir}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    dataset_name = 'grain'
    simplenet.set_model_dir(args.save_dir, dataset_name)

    if not args.test:
        i_auroc, p_auroc, pro_auroc = simplenet.train(train_dataloader, test_dataloader, args.with_mask)
    else:
        print("Testing...")
        i_auroc, p_auroc, pro_auroc = simplenet.test(train_dataloader, test_dataloader, args.save_segmentation_images)


    result_collect = []
    result_collect.append(
        {
            "dataset_name": dataset_name,
            "instance_auroc": i_auroc,  # auroc,
            "full_pixel_auroc": p_auroc,  # full_pixel_auroc,
            "anomaly_pixel_auroc": pro_auroc,
        }
    )

    for key, item in result_collect[-1].items():
        if key != "dataset_name":
            logger.info("{0}: {1:3.3f}".format(key, item))

    logger.info("\n\n-----\n")
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        args.save_dir,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


if __name__ == "__main__":
    main()