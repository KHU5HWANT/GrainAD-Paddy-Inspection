import os
from PIL import Image
import numpy as np
import random
import shutil
from pathlib import Path


if __name__ == '__main__':
    data_dir = '/mnt/lpai-dione/ssai/cvg/team/share_datasets/Grain/graindataset/sample5k'
    process_dir = os.path.join(data_dir, 'test', 'bad')
    image_list = os.listdir(process_dir)

    for img_path in image_list:
        image_path = os.path.join(process_dir, img_path)
        print(f'Processing: {image_path}')

        img = Image.open(image_path)
        width, height = img.size
        part_width = width // 5
        for i in range(5):
            left = i * part_width
            right = (i + 1) * part_width if i < 4 else width
            cropped_img = img.crop((left, 0, right, height))
            save_path = process_dir.replace('sample5k', 'sample_split')
            os.makedirs(save_path, exist_ok=True)
            cropped_img.save(os.path.join(save_path, f'{img_path.split(".")[0]}_{i}.png'))

        mask_path = process_dir.replace('bad', 'mask')
        mask_img = Image.open(os.path.join(mask_path, img_path))
        mask_width, mask_height = mask_img.size
        part_mask_width = mask_width // 5
        for i in range(5):
            left = i * part_mask_width
            right = (i + 1) * part_mask_width if i < 4 else mask_width
            cropped_mask = mask_img.crop((left, 0, right, mask_height))
            save_path = mask_path.replace('sample5k', 'sample_split')
            os.makedirs(save_path, exist_ok=True)
            cropped_mask.save(os.path.join(save_path, f'{img_path.split(".")[0]}_{i}_mask.png'))