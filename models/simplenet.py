"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.common as common
import utils.metrics as metrics
import cv2
import random
import time
from models.ICA import STNModule
from models.loss import SupConLoss, multiview_labels
from utils.training_state import save_training, load_training, pause_requested


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


LOGGER = logging.getLogger(__name__)


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:

    def __init__(self, log_dir):
        self.g_iter = 0
        LOGGER = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


def fusion_vis(image, mask):
    image = image.squeeze().permute(1,2,0).cpu().numpy()
    image = np.ascontiguousarray(image)
    mask = mask.squeeze()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ((image * std + mean)*255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alpha = np.zeros_like(image)
    # cv2.drawContours(alpha, contours, -1, (255,0,0), -1)
    cv2.drawContours(image, contours, -1, (255,50,0), 2)
    # image = cv2.addWeighted(image, 1, alpha, 0.3, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,  # 1536
            target_embed_dimension,  # 1536
            patchsize=3,  # 3
            patchstride=1,
            embedding_size=None,  # 256
            meta_epochs=1,  # 40
            aed_meta_epochs=1,
            gan_epochs=1,  # 4
            noise_std=0.05,
            mix_noise=1,
            noise_type="GAU",
            dsc_layers=2,  # 2
            dsc_hidden=None,  # 1024
            dsc_margin=.8,  # .5
            dsc_lr=0.0002,
            train_backbone=False,
            auto_noise=0,
            cos_lr=False,
            lr=1e-3,
            pre_proj=0,  # 1
            proj_layer_type=0,
            **kwargs,
    ):
        self.visualization = kwargs['visualization']
        self.visual_num = kwargs['visual_num']
        self.add_ica = kwargs['add_ica']
        self.five_split = kwargs['five_split']
        if self.add_ica:
            self.ica = STNModule(in_num=3, padding=0, mode='rotation').to(device)
            print(kwargs['ica_path'])
            self.ica.load_state_dict(torch.load(kwargs['ica_path'], map_location=device, weights_only=True))

        if self.add_ica:
            self.ica.requires_grad_(False)
        self.scl_weight = kwargs.get("scl_weight", 0.5)
        self.bce_weight = kwargs.get("bce_weight", 0.5)

        self.be_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.add_scl = kwargs['add_scl']
        if self.add_scl:
            self.scl_loss_fn = SupConLoss(temperature=0.1)
        self.ablation = kwargs['ablation']
        self.backbone = backbone.to(device)
        if not train_backbone:
            self.backbone.requires_grad_(False)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj,
                                             proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr=dsc_lr)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt,
                                                                   (meta_epochs - aed_meta_epochs) * gan_epochs,
                                                                   self.dsc_lr * .4)
        self.dsc_margin = dsc_margin

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None


    def set_model_dir(self, model_dir, dataset_name):
        self.save_path = os.path.join(model_dir, 'visualization')
        os.makedirs(self.save_path, exist_ok=True)
        self.model_dir = os.path.join(model_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features)  # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features)  # further pooling

        return features, patch_shapes

    def test(self, training_data, test_data, save_segmentation_images):

        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            print('** exist ,', ckpt_path)
            # state_dicts = torch.load(ckpt_path, map_location=self.device)
            # if "pretrained_enc" in state_dicts:
            #     self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            # if "pretrained_dec" in state_dicts:
            #     self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])
        state_dict = torch.load(ckpt_path, map_location=self.device)
        if 'discriminator' in state_dict:
            self.discriminator.load_state_dict(state_dict['discriminator'])
            if "pre_projection" in state_dict:
                self.pre_projection.load_state_dict(state_dict["pre_projection"])
        else:
            self.load_state_dict(state_dict, strict=False)

        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        return self._evaluate(test_data, scores, segmentations, features, labels_gt,
                              masks_gt, getattr(test_data.dataset, 'with_mask', False))

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt, with_mask=True):

        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / np.maximum(img_max_scores - img_min_scores, 1e-12)
        # scores = np.mean(scores, axis=0)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]

        if len(masks_gt) > 0 and with_mask:
            segmentations = np.array(segmentations)

            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt)
            # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            # pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
            #                           norm_segmentations)
            pro = -1
        else:
            full_pixel_auroc = -1
            pro = -1

        return auroc, full_pixel_auroc, pro

    def train(self, training_data, test_data, with_mask=True):

        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")

        # if os.path.exists(ckpt_path):
        #     state_dict = torch.load(ckpt_path, map_location=self.device)
        #     if 'discriminator' in state_dict:
        #         self.discriminator.load_state_dict(state_dict['discriminator'])
        #         if "pre_projection" in state_dict:
        #             self.pre_projection.load_state_dict(state_dict["pre_projection"])
        #     else:
        #         self.load_state_dict(state_dict, strict=False)
        #
        #     self.predict(training_data, "train_")
        #     scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        #     auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features,
        #                                                                   labels_gt, masks_gt)
        #
        #     return auroc, full_pixel_auroc, anomaly_pixel_auroc

        def update_state_dict(d):

            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        best_record = None
        resume_path = os.path.join(self.ckpt_dir, 'training_state.pth')
        first_epoch = 0
        last_metrics = [-1, -1, -1]
        if os.path.exists(resume_path):
            state = load_training(resume_path, self.device)
            self.discriminator.load_state_dict(state['discriminator'])
            self.dsc_opt.load_state_dict(state['dsc_opt'])
            self.dsc_schl.load_state_dict(state['scheduler'])
            if self.pre_proj > 0:
                self.pre_projection.load_state_dict(state['pre_projection'])
                self.proj_opt.load_state_dict(state['proj_opt'])
            if self.train_backbone:
                self.backbone.load_state_dict(state['backbone'])
                self.backbone_opt.load_state_dict(state['backbone_opt'])
            first_epoch = state['epoch']
            best_record = state['best_record']
            last_metrics = state['metrics']
            LOGGER.info(f'Resuming GrainAD at epoch {first_epoch + 1}')
        for i_mepoch in range(first_epoch, self.meta_epochs):

            self._train_discriminator(training_data)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt,
                                                          masks_gt, with_mask)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
                # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")

            # Save the final epoch independently of held-out test metrics.
            update_state_dict(state_dict)
            torch.save(state_dict, ckpt_path)
            last_metrics = [auroc, full_pixel_auroc, pro]
            resume = dict(epoch=i_mepoch + 1, total_epochs=self.meta_epochs,
                          discriminator=self.discriminator.state_dict(), dsc_opt=self.dsc_opt.state_dict(),
                          scheduler=self.dsc_schl.state_dict(), best_record=best_record, metrics=last_metrics)
            if self.pre_proj > 0:
                resume.update(pre_projection=self.pre_projection.state_dict(), proj_opt=self.proj_opt.state_dict())
            if self.train_backbone:
                resume.update(backbone=self.backbone.state_dict(), backbone_opt=self.backbone_opt.state_dict())
            save_training(resume_path, **resume)
            if pause_requested():
                LOGGER.info('Saved training state; paused at epoch boundary.')
                raise SystemExit(75)

        return last_metrics

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        if self.add_ica:
            self.ica.eval()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                all_p_sync = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = torch.cat(img, dim=0)
                    img = img.to(torch.float).to(self.device)
                    aug_img = data_item["aug_image"]
                    masks = data_item["aug_mask"]
                    if self.five_split is True:
                        aug_img = torch.cat(aug_img, dim=0)
                        masks = torch.cat(masks, dim=0)
                    else:
                        aug_img = torch.cat([aug_img], dim=0)
                        masks = torch.cat([masks], dim=0)

                    aug_img = aug_img.to(torch.float).to(self.device)
                    masks = masks.to(self.device)
                    aug_labels = data_item["aug_labels"]
                    if self.five_split:
                        aug_labels = aug_labels.transpose(0, 1)
                    aug_labels = aug_labels.reshape(-1).to(self.device)
                    if self.add_ica:
                        img, _ = self.ica(img)
                        aug_img, theta = self.ica(aug_img)
                        grid = F.affine_grid(theta, masks.shape, align_corners=False)
                        masks = F.grid_sample(masks, grid, mode='nearest', padding_mode='border', align_corners=False)

                    true_feats, patch_shapes = self._embed(img, evaluation=False)
                    sync_feats, patch_shapes = self._embed(aug_img, evaluation=False)
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(true_feats)
                        sync_feats = self.pre_projection(sync_feats)

                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(
                        self.device)  # (N, K)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1 ** (k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device)  # (N, K, C)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    fake_feats = true_feats + noise
                    if self.ablation == 0 or self.ablation == 3:
                        scores = self.discriminator(torch.cat([sync_feats, true_feats, fake_feats]))
                        score_gap = len(sync_feats)
                        sync_scores = scores[:score_gap]
                        true_scores = scores[score_gap:2 * score_gap]
                        fake_scores = scores[2 * score_gap:]
                        th = self.dsc_margin
                        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                        p_sync = (sync_scores.detach() < -th).sum() / len(sync_scores)
                        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                        scales = patch_shapes[0]
                        pred = torch.cat([true_scores, fake_scores, sync_scores], dim=1)
                        pred = pred.reshape(-1, scales[0], scales[1], 3)
                        pred = torch.permute(pred, (0, 3, 1, 2))
                        gt_label = torch.zeros_like(pred, dtype=torch.float32).to(self.device)
                        resize_mask = F.interpolate(masks.float(), pred.shape[2:], mode='nearest').squeeze(1)
                        gt_label[:, 0, :, :] = 1
                        gt_label[:, 2, :, :] = 1 - resize_mask
                        loss = self.be_loss_fn(pred, gt_label)
                        if self.ablation == 3:
                            self.add_scl = False
                        else:
                            self.add_scl = True

                    elif self.ablation == 1:
                        scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                        true_scores = scores[:len(true_feats)]
                        fake_scores = scores[len(fake_feats):]
                        th = self.dsc_margin
                        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                        true_loss = torch.clip(-true_scores + th, min=0)
                        fake_loss = torch.clip(fake_scores + th, min=0)
                        loss = true_loss.mean() + fake_loss.mean()
                        self.add_scl = False

                    elif self.ablation == 2:
                        scores = self.discriminator(torch.cat([sync_feats, true_feats]))
                        score_gap = len(sync_feats)
                        sync_scores = scores[:score_gap]
                        true_scores = scores[score_gap:]
                        th = self.dsc_margin
                        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                        p_sync = (sync_scores.detach() < -th).sum() / len(sync_scores)
                        scales = patch_shapes[0]
                        pred = torch.cat([true_scores, sync_scores], dim=1)
                        pred = pred.reshape(-1, scales[0], scales[1], 2)
                        pred = torch.permute(pred, (0, 3, 1, 2))
                        gt_label = torch.zeros_like(pred, dtype=torch.float32).to(self.device)
                        resize_mask = F.interpolate(masks.float(), pred.shape[2:], mode='nearest').squeeze(1)
                        gt_label[:, 0, :, :] = 1
                        gt_label[:, 1, :, :] = 1 - resize_mask
                        loss = self.be_loss_fn(pred, gt_label)
                        self.add_scl = False
                    else:
                        raise ValueError(f"Invalid ablation number: {self.ablation}")

                    if self.add_scl:
                        b = gt_label.shape[0]
                        true_feats = true_feats.reshape(b, -1)
                        fake_feats = fake_feats.reshape(b, -1)
                        sync_feats = sync_feats.reshape(b, -1)

                        all_feats = torch.cat([true_feats, fake_feats, sync_feats], dim=0)
                        num_views = 5 if self.five_split else 1
                        labels = multiview_labels(b // num_views, num_views, aug_labels)
                        loss = (self.bce_weight * loss + self.scl_weight *
                                self.scl_loss_fn(all_feats, labels))

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    if self.ablation != 2:
                        all_p_fake.append(p_fake.cpu().item())
                    if self.ablation != 1:
                        all_p_sync.append(p_sync.cpu().item())

                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

                if self.cos_lr:
                    self.dsc_schl.step()

                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data) if self.ablation != 2 else 0
                all_p_sync = sum(all_p_sync) / len(input_data) if self.ablation != 1 else 0
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)} p_sync:{round(all_p_sync, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            img_num = 50
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].cpu().numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats, align_img, theta_list = self._predict(image)

                # Image-only AUROC needs scores, not thousands of full-resolution maps.
                collect_maps = getattr(dataloader.dataset, 'with_mask', True) or self.visualization
                if collect_maps:
                    if self.add_ica:
                        gt_list = []
                        for gt, theta in zip(data["mask"], theta_list):
                            gt = gt.to(theta.device)
                            grid = F.affine_grid(theta, gt.shape, align_corners=False)
                            gt = F.grid_sample(gt, grid, mode='nearest', padding_mode='border', align_corners=False)
                            gt_list.append(gt)
                    else:
                        gt_list = data["mask"]
                    gt = torch.cat(gt_list, dim=3)
                    masks_gt.extend(gt.cpu().numpy())

                for score, mask, is_anomaly in zip(_scores, _masks, data["is_anomaly"].cpu().numpy().tolist()):
                    scores.append(score)
                    if collect_maps:
                        masks.append(mask)
                if self.visualization:
                    if self.visual_num > 0:
                        image = torch.cat(image, dim=3)
                        pred = masks[-1]
                        pred = (pred - pred.min()) / (pred.max() - pred.min())
                        # thresholdize
                        threshold = 0.6
                        highfactor = 1.2
                        lowfactor = 0.5

                        pred[pred >= threshold] *= highfactor
                        pred[pred < threshold] *= lowfactor
                        pred = (pred - pred.min()) / (pred.max() - pred.min())


                        heatmap = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
                        heatmap = heatmap[..., ::-1]

                        mask = masks_gt[-1]

                        fusioned_img = fusion_vis(image, mask)

                        mask = np.repeat(mask, 3, axis=0)
                        mask = mask.squeeze().transpose(1, 2, 0)
                        mask = (mask * 255).astype(np.uint8)

                        mean = np.array([[0.485, 0.456, 0.406] for ix in range(len(image))])
                        std = np.array([[0.229, 0.224, 0.225] for ix in range(len(image))])

                        ximg = image.cpu().numpy() * std[:, :, None, None] + mean[:, :, None, None]
                        align_img = align_img.cpu().numpy() * std[:, :, None, None] + mean[:, :, None, None]
                        align_img = np.transpose(align_img, (0, 2, 3, 1))
                        align_img = (align_img[0] * 255).astype(np.uint8)
                        align_img = cv2.addWeighted(align_img, 0.5, heatmap, 0.5, 0)

                        fimg = np.transpose(ximg, (0, 2, 3, 1))
                        fimg = (fimg[0] * 255).astype(np.uint8)
                        show_img = np.concatenate([fimg, align_img], axis=0)
                        show_img = show_img[..., ::-1]

                        # cv2.imwrite(f'img_{self.visual_num}.png', mask)

                        save_img = np.vstack([show_img, mask])
                        align_img = align_img[..., ::-1]
                        # cv2.imwrite(os.path.join(self.save_path, f'img_{self.visual_num}.png'), save_img)
                        cv2.imwrite(os.path.join(self.save_path, f'img_{self.visual_num}_gt.png'), fusioned_img)
                        cv2.imwrite(os.path.join(self.save_path, f'img_{self.visual_num}_pred.png'), align_img)

                        self.visual_num -= 1

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        _ = self.forward_modules.eval()

        batchsize = images[0].shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        if self.add_ica:
            self.ica.eval()
        self.discriminator.eval()
        masks_list = []
        scores_list = []
        align_images = []
        theta_list = []
        with torch.no_grad():
            for i, batch in enumerate(images):
                if self.add_ica:
                    batch, theta = self.ica(batch)
                    theta_list.append(theta)

                align_images.append(batch)

                features, patch_shapes = self._embed(batch,
                                                     provide_patch_shapes=True,
                                                     evaluation=True)
                if self.pre_proj > 0:
                    features = self.pre_projection(features)

                # features = features.cpu().numpy()
                # features = np.ascontiguousarray(features.cpu().numpy())
                patch_scores = image_scores = -self.discriminator(features)
                patch_scores = patch_scores.cpu().numpy()
                image_scores = image_scores.cpu().numpy()

                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                )
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
                image_scores = self.patch_maker.score(image_scores)
                scores_list.append(image_scores)
                patch_scores = self.patch_maker.unpatch_scores(
                    patch_scores, batchsize=batchsize
                )
                scales = patch_shapes[0]
                patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
                features = features.reshape(batchsize, scales[0], scales[1], -1)
                masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
                masks_list.append(masks)

        masks = np.concatenate(masks_list, axis=2)
        image_scores = np.stack(scores_list, axis=0).max(axis=0)
        align_images = torch.cat(align_images, dim=3)

        return list(image_scores), list(masks), list(features), align_images, theta_list

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()
