import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil
from glob import glob


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train', split=None):
        
        dataset_dir = args.dataset_dir

        if mode not in ('train', 'test'):
            raise ValueError('Unknown dataset mode: {}'.format(mode))

        # Data augmentation (``mode``) and manifest selection (``split``) are
        # deliberately independent.  Only the official train/test protocol is
        # accepted; there is no validation split in this experiment.
        self.split = split or mode
        if self.split not in ('train', 'test'):
            raise ValueError(
                'Unsupported dataset split: {}. Only official train/test '
                'splits are allowed.'.format(self.split)
            )
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')
        manifests = self._validate_data_contract(dataset_dir)
        self.list_dir, self.names = manifests[self.split]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = osp.splitext(self.names[i])[0]
        img_path = self._resolve_image_path(self.imgs_dir, name)
        label_path = self._resolve_image_path(self.label_dir, name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError('Unknown dataset mode: {}'.format(self.mode))

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return {
            'image': img,
            'mask': mask,
            'name': name,
        }

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _find_split_file(dataset_dir, split):
        if split not in ('train', 'test'):
            raise ValueError(
                'Unsupported dataset split: {}. Only official train/test '
                'splits are allowed.'.format(split)
            )
        dataset_name = osp.basename(osp.normpath(dataset_dir))
        path = osp.join(
            dataset_dir,
            'img_idx',
            '{}_{}.txt'.format(split, dataset_name),
        )
        if not osp.isfile(path):
            raise FileNotFoundError(
                'Missing official {} split manifest: {}'.format(split, path)
            )
        return path

    @classmethod
    def _validate_data_contract(cls, dataset_dir):
        """Validate the immutable official train/test dataset protocol."""

        manifests = {}
        name_sets = {}
        for split in ('train', 'test'):
            path = cls._find_split_file(dataset_dir, split)
            names = cls._read_split_names(path)
            if not names:
                raise ValueError('{} split manifest is empty: {}'.format(split, path))
            duplicates = cls._duplicates(names)
            if duplicates:
                raise ValueError(
                    'Duplicate image names in {} split manifest {}: {}'.format(
                        split, path, ', '.join(sorted(duplicates)[:10])
                    )
                )
            canonical_names = [cls._canonical_name(name) for name in names]
            manifests[split] = (path, canonical_names)
            name_sets[split] = set(canonical_names)

        overlap = name_sets['train'] & name_sets['test']
        if overlap:
            raise ValueError(
                'Dataset split leakage between official train and test: {}'.format(
                    ', '.join(sorted(overlap)[:10])
                )
            )

        images_dir = osp.join(dataset_dir, 'images')
        masks_dir = osp.join(dataset_dir, 'masks')
        for split in ('train', 'test'):
            for name in manifests[split][1]:
                try:
                    cls._resolve_image_path(images_dir, name)
                except FileNotFoundError as error:
                    raise FileNotFoundError(
                        'Missing image referenced by {} split: {}'.format(split, name)
                    ) from error
                try:
                    cls._resolve_image_path(masks_dir, name)
                except FileNotFoundError as error:
                    raise FileNotFoundError(
                        'Missing mask referenced by {} split: {}'.format(split, name)
                    ) from error
        return manifests

    @staticmethod
    def _read_split_names(path):
        with open(path, 'r', encoding='utf-8') as handle:
            return [line.strip() for line in handle if line.strip()]

    @staticmethod
    def _canonical_name(name):
        return osp.splitext(name.strip().replace('\\', '/'))[0]

    @classmethod
    def _duplicates(cls, names):
        seen = set()
        duplicates = set()
        for name in names:
            canonical = cls._canonical_name(name)
            if canonical in seen:
                duplicates.add(canonical)
            seen.add(canonical)
        return duplicates

    @staticmethod
    def _resolve_image_path(root, name):
        path = osp.join(root, name + '.png')
        if osp.exists(path):
            return path

        matches = sorted(glob(osp.join(root, name + '.*')))
        if matches:
            return matches[0]

        raise FileNotFoundError('Cannot find image/mask for "{}" under {}'.format(name, root))

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
