import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from data.mvtec import MVTecDataset, DatasetSplit
from torchvision import transforms
from models.loss import SupConLoss
import warnings
import ssl
from data.grain import GrainDataset, initialize_worker
import utils.utils as utils
from models.ICA import STNModule
from tqdm import tqdm
from utils.training_state import save_training, load_training, pause_requested


warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def visualize_stn(stn, test_loader, train_loader, args, num_images=20):
    logger.info("Visualizing...")
    stn.eval()
    save_path = os.path.join(args.save_dir, 'visualization')
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i == num_images:
                break
            img = batch['image']
            aug_img = batch['aug_image']
            mask = batch['aug_mask']
            img = torch.cat(img, dim=0)
            aug_img = torch.cat(aug_img, dim=0)
            mask = torch.cat(mask, dim=0)
            z, theta = stn(aug_img)
            grid = F.affine_grid(theta, torch.Size(mask.shape))
            mask = F.grid_sample(mask, grid, padding_mode='border')

            mean = np.array([[0.485, 0.456, 0.406] for ix in range(len(img))])
            std = np.array([[0.229, 0.224, 0.225] for ix in range(len(img))])
            img = img.cpu().numpy() * std[:, :, None, None] + mean[:, :, None, None]
            img1 = z.cpu().numpy() * std[:, :, None, None] + mean[:, :, None, None]
            img = np.transpose(img, (0, 2, 3, 1))
            img1 = np.transpose(img1, (0, 2, 3, 1))
            show_img1 = np.concatenate((img, img1), axis=1)

            aug_img = aug_img.cpu().numpy() * std[:, :, None, None] + mean[:, :, None, None]
            aug_img = np.transpose(aug_img, (0, 2, 3, 1))
            mask = mask.cpu().numpy()
            mask = np.repeat(mask, 3, axis=1)
            mask = np.transpose(mask, (0, 2, 3, 1))
            show_img2 = np.concatenate((aug_img, mask), axis=1)

            show_img = np.hstack((show_img1, show_img2))

            show_img = show_img[..., ::-1]

            cv2.imwrite(os.path.join(save_path, f'img_{i}.png'), (show_img[0] * 255).astype(np.uint8))

@hydra.main(config_path="../configs", config_name="train_ica", version_base=None)
def main(args):
    set_seed(42)
    args = DictConfig(OmegaConf.to_container(args, resolve=True))
    logger.info(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print(accelerator.device)
    # dataset
    # data_path = r'/mnt/lpai-dione/ssai/cvg/team/share_datasets/MVTec AD'
    # train_dataset = MVTecDataset(source=data_path, classname='screw', split=DatasetSplit.TRAIN)
    # test_dataset = MVTecDataset(source=data_path, classname='screw', split=DatasetSplit.TEST)
    train_dataset = GrainDataset(args.train_data_dir, five_split=True, **args)
    test_dataset = GrainDataset(args.train_data_dir, is_train=False, five_split=True, **args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=initialize_worker, persistent_workers=args.num_workers > 0)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)

    # model
    stn = STNModule(3, padding=0, mode="rotation")
    train_params = [p for p in stn.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(train_params, lr=args.lr)

    stn, train_dataloader, test_dataloader, optimizer = accelerator.prepare(
        stn, train_dataloader, test_dataloader, optimizer
    )

    # ------------------------------ TRAIN ------------------------------ #
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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

    global_step = 0
    first_epoch = 0
    resume_path = os.path.join(args.save_dir, 'models', 'training_state.pth')
    if os.path.exists(resume_path):
        state = load_training(resume_path, accelerator.device)
        accelerator.unwrap_model(stn).load_state_dict(state['model'])
        if state.get('optimizer') is not None:
            optimizer.load_state_dict(state['optimizer'])
        first_epoch = state['epoch']
        global_step = first_epoch * num_update_steps_per_epoch
        logger.info(f'Resuming alignment at epoch {first_epoch + 1}')

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        stn.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(stn):
                target = batch['image']
                target = torch.cat(target, dim=0)
                img1 = target.clone()
                # Select another view of the SAME kernel, preserving view-major order.
                views = torch.stack(batch['image'], dim=0)
                order = torch.randperm(len(views), device=views.device)
                img2 = views[order].flatten(0, 1)

                z1, _ = stn(img1)
                z2, _ = stn(img2)
                loss = F.mse_loss(z1, z2)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0 or epoch == args.num_train_epochs - 1:
                stn = accelerator.unwrap_model(stn)
                save_path = os.path.join(args.save_dir, 'models')
                os.makedirs(save_path, exist_ok=True)
                torch.save(stn.state_dict(), os.path.join(save_path, f"ica.pth"))
            save_training(resume_path, model=accelerator.unwrap_model(stn).state_dict(),
                          optimizer=optimizer.state_dict(), epoch=epoch + 1,
                          total_epochs=args.num_train_epochs)
            if pause_requested():
                logger.info('Saved training state; paused at epoch boundary.')
                raise SystemExit(75)
            if epoch % args.validation_epochs == 0:
                visualize_stn(stn, test_dataloader, train_dataloader, args)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
