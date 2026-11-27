from enum import Enum
from glob import glob
import os
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
import imgaug.augmenters as iaa
import cv2
from PIL import ImageOps
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.perlin import rand_perlin_2d_np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


data_path = '/mnt/lpai-dione/ssai/cvg/team/share_datasets/Grain/graindataset'


class GrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=data_path, classname='soybean', five_split=False, is_train=True, with_mask=False, **kwargs):
        self.data_path = data_path
        self.resize_shape = kwargs['resize']
        self.padding = kwargs['padding']
        self.with_mask = with_mask
        self.generate_anomalies = kwargs.get("generate_anomalies", True)

        # self.train_data = glob(os.path.join(data_path, classname, 'train', 'good', '*.png'))
        self.train_data = sorted(glob(os.path.join(data_path, classname, 'train', 'good', '*.png')))
        self.train_labels = [0] * len(self.train_data)
        print('train length: ', len(self.train_data))

        self.test_folder = 'test-pixel' if self.with_mask else 'test-image'
        # self.test_good_data = glob(os.path.join(data_path, classname, self.test_folder, 'good', '*.png'))
        # self.test_bad_data = glob(os.path.join(data_path, classname, self.test_folder, 'bad', '*.png'))
        self.test_good_data = sorted(glob(os.path.join(data_path, classname, self.test_folder, 'good', '*.png')))
        self.test_bad_data = sorted(glob(os.path.join(data_path, classname, self.test_folder, 'bad', '*.png')))
        self.test_data = self.test_bad_data + self.test_good_data
        self.test_labels = [1] * len(self.test_bad_data) + [0] * len(self.test_good_data)
        print('test length: ', len(self.test_good_data), len(self.test_bad_data))


        if not (self.train_data if is_train else self.test_data):
            raise ValueError(f'No PNG images found for {classname} in {data_path}')
        self.is_train = is_train

        self.five_split = five_split
        self.data_to_iterate = self.get_image_data()
        self.resize = kwargs['resize']
        if self.five_split:
            self.transform_img = [
                transforms.Resize(kwargs['resize']),
                # transforms.CenterCrop(kwargs['imagesize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img = transforms.Compose(self.transform_img)

            self.transform_mask = [
                transforms.Resize(kwargs['resize'], interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
            self.transform_mask = transforms.Compose(self.transform_mask)
            self.imagesize = (3, kwargs['imagesize'][0], kwargs['imagesize'][1])
        else:
            self.transform_img = [
                # transforms.Resize(kwargs['resize']),
                # transforms.CenterCrop(kwargs['imagesize']),
                transforms.Resize((kwargs['imagesize'][0], kwargs['imagesize'][1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img = transforms.Compose(self.transform_img)

            self.transform_mask = [
                # transforms.Resize(kwargs['resize']),
                # transforms.CenterCrop(kwargs['imagesize']),
                transforms.Resize((kwargs['imagesize'][0], kwargs['imagesize'][1])),
                transforms.ToTensor(),
            ]
            self.transform_mask = transforms.Compose(self.transform_mask)
            self.imagesize = (3, kwargs['imagesize'][0], kwargs['imagesize'][1])

        # anomaly generator
        self.anomaly_source_paths = sorted(glob(kwargs['anomaly_source_path'] + "/*/*.jpg"))
        if is_train and self.generate_anomalies and not self.anomaly_source_paths:
            raise ValueError('Training requires DTD textures at anomaly_source_path/*/*.jpg')
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def __len__(self):
        if self.is_train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        augmented_image = []
        anomaly_mask = []
        aug_labels = []
        if self.is_train:
            image_path = self.train_data[index]
            if self.five_split:
                image, mask, augmented_image, anomaly_mask, aug_labels = self.crop_image_into_5_parts(image_path)
            else:
                image = Image.open(image_path).convert("RGB")
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item() if self.generate_anomalies else None
                image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(
                    image, self.anomaly_source_paths[anomaly_source_idx] if anomaly_source_idx is not None else None
                )
                aug_labels = np.concatenate([has_anomaly])
                if self.padding != 0:
                    image = ImageOps.expand(image, border=self.padding, fill='black')
                    augmented_image = ImageOps.expand(augmented_image, border=self.padding, fill='black')
                    anomaly_mask = ImageOps.expand(anomaly_mask, border=self.padding, fill='black')

                image = self.transform_img(image)
                augmented_image = self.transform_img(augmented_image)
                anomaly_mask = self.transform_mask(anomaly_mask)
                mask = torch.zeros([1, *image.size()[1:]])
                mask = [mask]
                image = [image]
            target = self.train_labels[index]
            anomaly = 'good'
        else:
            image_path = self.test_data[index]
            augmented_image = []
            anomaly_mask = []
            if self.five_split:
                image, mask, _, _, _ = self.crop_image_into_5_parts(image_path)
            else:
                image = Image.open(image_path).convert("RGB")

                if self.padding != 0:
                    image = image.resize((self.resize_shape[1], self.resize_shape[0]))
                    image = ImageOps.expand(image, border=self.padding, fill='black')

                image = self.transform_img(image)
                mask_path = image_path.replace('bad', 'mask')
                if os.path.exists(mask_path) and not 'good' in image_path:
                    mask = Image.open(mask_path).convert("L")
                    if self.padding != 0:
                        mask = mask.resize((self.resize_shape[1], self.resize_shape[0]))
                        mask = ImageOps.expand(mask, border=self.padding, fill='black')
                    mask = self.transform_mask(mask)
                else:
                    mask = torch.zeros([1, *image.size()[1:]])
                mask = [mask]
                image = [image]
            target = self.test_labels[index]
            anomaly = 'bad' if target == 1 else 'good'

        return {
            'image': image,
            'aug_image': augmented_image,
            'aug_mask': anomaly_mask,
            'aug_labels': aug_labels,
            'is_anomaly': target,
            'mask': mask,
            'classname': 'grain',
            "image_name": "/".join(image_path.split("/")[-4:]),
            'anomaly': anomaly,
            "image_path": image_path
        }

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def get_image_data(self):
        data_to_iterate = []
        if self.is_train:
            for anomaly, img_path in zip(self.train_labels, self.train_data):
                data_to_iterate.append(('', anomaly, img_path, None))
        else:
            for anomaly, img_path in zip(self.test_labels, self.test_data):
                if anomaly == 0:
                    data_to_iterate.append(('', anomaly, img_path, None))
                else:
                    mask_path = img_path.replace('bad', 'mask')
                    data_to_iterate.append(('', anomaly, img_path, mask_path))

        return data_to_iterate

    def transform_image(self, pil_image, anomaly_source_path):
        if not self.generate_anomalies:
            image = pil_image.resize((self.resize_shape[1], self.resize_shape[0]))
            return image, image.copy(), Image.new('L', image.size), np.array([0.0], dtype=np.float32)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        anomaly_mask = anomaly_mask * 255.0
        anomaly_mask = anomaly_mask.astype(np.uint8)
        image = Image.fromarray(image[..., ::-1].astype(np.uint8))
        augmented_image = Image.fromarray(augmented_image[..., ::-1].astype(np.uint8))
        anomaly_mask = Image.fromarray(anomaly_mask.squeeze(), mode='L')

        return image, augmented_image, anomaly_mask, has_anomaly

    def crop_image_into_5_parts(self, image_path):
        """
        Crop the image into 5 equal parts based on width and return them in a list.
        """
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        part_width = width // 5
        cropped_images = []
        augmented_images = []
        anomaly_masks = []
        labels = []
        for i in range(5):
            left = i * part_width
            right = (i + 1) * part_width if i < 4 else width
            cropped_img = img.crop((left, 0, right, height))

            if self.is_train:
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item() if self.generate_anomalies else None
                cropped_img, augmented_img, anomaly_mask, has_anomaly = self.transform_image(
                    cropped_img, self.anomaly_source_paths[anomaly_source_idx] if anomaly_source_idx is not None else None
                )
                if self.padding != 0:
                    augmented_img = ImageOps.expand(augmented_img, border=self.padding, fill='black')
                    anomaly_mask = ImageOps.expand(anomaly_mask, border=self.padding, fill='black')

                augmented_img = self.transform_img(augmented_img)
                anomaly_mask = self.transform_mask(anomaly_mask)
                augmented_images.append(augmented_img)
                anomaly_masks.append(anomaly_mask)
                labels.append(has_anomaly)

            if self.padding != 0:
                cropped_img = cropped_img.resize((self.resize_shape[1], self.resize_shape[0]))
                cropped_img = ImageOps.expand(cropped_img, border=self.padding, fill='black')

            cropped_image = self.transform_img(cropped_img)
            cropped_images.append(cropped_image)
            del cropped_img

        cropped_masks = []
        mask_path = image_path.replace('bad', 'mask')
        if os.path.exists(mask_path) and 'good' not in image_path:
            msk = Image.open(mask_path).convert("L")
            for i in range(5):
                left = i * part_width
                right = (i + 1) * part_width if i < 4 else width
                cropped_msk = msk.crop((left, 0, right, height))
                if self.padding != 0:
                    cropped_msk = cropped_msk.resize((self.resize_shape[1], self.resize_shape[0]))
                    cropped_msk = ImageOps.expand(cropped_msk, border=self.padding, fill='black')
                cropped_mask = self.transform_mask(cropped_msk)
                cropped_masks.append(cropped_mask)
                del cropped_msk
        else:
            cropped_masks = [torch.zeros([1, *self.resize]) for _ in range(5)]

        if self.is_train:
            labels = np.concatenate(labels, axis=0)

        return cropped_images, cropped_masks, augmented_images, anomaly_masks, labels


def initialize_worker(worker_id):
    """Initialize imgaug's process-local cache and seed on Windows spawn."""
    import imgaug
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    imgaug.seed(seed)
    iaa.AddToHueAndSaturation(0)
    cv2.setNumThreads(1)
