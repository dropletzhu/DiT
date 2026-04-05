# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["ImageNetDataset", "ImageNetLatentDataset"]


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageNetDataset:
    """ImageNet dataset for DiT training.

    Args:
        dataset_path: Path to ImageNet root directory containing train/val folders
        image_size: Target image size (height and width)
        split: Dataset split ("train" or "val")
        class_dropout_prob: Probability of dropping class label for classifier-free guidance
    """

    output_columns = ["image", "class_label"]

    def __init__(
        self,
        dataset_path: str,
        image_size: int = 256,
        split: str = "train",
        class_dropout_prob: float = 0.1,
        latent_channels: int = 4,
        latent_path: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.split = split
        self.class_dropout_prob = class_dropout_prob
        self.latent_channels = latent_channels
        self.latent_path = latent_path

        self.samples = self._collect_samples()
        self.num_classes = 1000

    def _collect_samples(self) -> List[Tuple[str, int]]:
        samples = []
        split_folder = self.dataset_path / self.split

        if not split_folder.exists():
            raise ValueError(f"Dataset split folder not found: {split_folder}")

        for class_dir in sorted(split_folder.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if not class_name.startswith("n"):
                continue

            try:
                class_idx = int(class_name.split("n")[1].split("_")[0])
            except (IndexError, ValueError):
                continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in IMAGE_EXT:
                    samples.append((str(img_file), class_idx))

        if len(samples) == 0:
            raise ValueError(f"No images found in {split_folder}")

        return samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path, class_label = self.samples[idx]

        try:
            img = self._load_image(img_path)
        except Exception:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        return (img, class_label)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def train_transforms(image_size: int = 256, latent_channels: int = 4) -> List[Dict]:
        from mindspore.dataset.vision import ToTensor

        transforms = [
            {
                "operations": [ToTensor()],
                "input_columns": ["image"],
                "output_columns": ["image"],
            },
        ]
        return transforms

    @staticmethod
    def collate_fn(samples: List[Tuple[Any, ...]]) -> Dict[str, Any]:
        import mindspore as ms

        images = np.stack([s[0] for s in samples])
        images = images.astype(np.float32) / 255.0
        images = images.transpose(0, 3, 1, 2)
        images = images * 2.0 - 1.0
        class_labels = [s[1] for s in samples]

        return {
            "image": ms.Tensor.from_numpy(images),
            "class_label": ms.Tensor(class_labels, dtype=ms.int32),
        }


class ImageNetLatentDataset:
    """ImageNet dataset with pre-computed VAE latents for faster training.

    Args:
        latent_path: Path to pre-computed latents folder
        class_dropout_prob: Probability of dropping class label for classifier-free guidance
    """

    output_columns = ["latent", "class_label"]

    def __init__(
        self,
        latent_path: str,
        class_dropout_prob: float = 0.1,
        latent_channels: int = 4,
        sample_size: int = 32,
    ):
        super().__init__()
        self.latent_path = Path(latent_path)
        self.class_dropout_prob = class_dropout_prob
        self.latent_channels = latent_channels
        self.sample_size = sample_size

        self.samples = self._collect_samples()
        self.num_classes = 1000

    def _collect_samples(self) -> List[Tuple[str, int]]:
        samples = []

        for class_dir in sorted(self.latent_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if not class_name.startswith("n"):
                continue

            try:
                class_idx = int(class_name.split("n")[1].split("_")[0])
            except (IndexError, ValueError):
                continue

            for latent_file in class_dir.iterdir():
                if latent_file.suffix == ".npy":
                    samples.append((str(latent_file), class_idx))

        if len(samples) == 0:
            raise ValueError(f"No latents found in {self.latent_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        latent_path, class_label = self.samples[idx]

        try:
            latent = np.load(latent_path)
        except Exception:
            latent = np.zeros((self.latent_channels, self.sample_size, self.sample_size), dtype=np.float32)

        return latent, class_label

    @staticmethod
    def train_transforms(latent_channels: int = 4) -> List[Dict]:
        return []

    @staticmethod
    def collate_fn(samples: List[Tuple[Any, ...]]) -> Dict[str, Any]:
        import mindspore as ms

        latents = np.stack([s[0] for s in samples])
        class_labels = [s[1] for s in samples]

        return {
            "latent": ms.Tensor.from_numpy(latents.astype(np.float32)),
            "class_label": ms.Tensor(class_labels, dtype=ms.int32),
        }