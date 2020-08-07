"""
Returns the data and performs transformations if in train mode. If the image is flipped horizontally, the target and condition are 
also flipped accordingly. 

Transform: The normalization is done to fit pytorches pretrained vgg model.
    During training a small random crop is performed 50% of the time for more data augmentation.
Target transform: The targets are transformed to a one hot encoding
Condition transform: The conditions are transformed to a one hot encoding
"""
from torch.utils import data
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
import numpy as np
import random
import os
from glob import glob
import imgaug.augmenters as iaa
from imgaug import parameters as iap
from pathlib import Path
from typing import Tuple

class Dataset(data.Dataset):
    def __init__(self, data_path: Path, train: bool = True):
        """Inits the dataset

        Args:
            data_path (Path): path to the data
            train (bool, optional): True if the data is loaded for training. Defaults to True.
        """        
        self.samples = []
        self.data_path = data_path
        self.train = train #for data augmentation and drop last
        self.target_dict = {} 
        self.make_dataset()

    def __len__(self) -> int:
        """Returns the size of data

        Returns:
            int: Size of data
        """         
        return len(self.samples)

    def transform(self, image: np.ndarray, target: str, condition: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Transforms and normalizes the data. If in training mode the data is augmentated.

        Args:
            image (np.ndarray): Image to transform
            target (str): Training target
            condition (int): Condition

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: Augmented image, target and condition
        """                 
        # Resize
        resize = iaa.Resize({"height": 224, "width": 224})
        image = resize.augment_image(image)

        # Random horizontal flipping and erase
        if self.train: 

            if random.random() > 0.5: 
                # flip image
                flip = iaa.HorizontalFlip(1.0)
                image = flip.augment_image(image)

                # flip class
                if target == "a":
                    target = "d"
                elif target == "d":
                    target = "a"

                # flip condition 
                if condition == 2:
                    condition = 4
                elif condition == 4:
                    condition = 2

            #imgaug
            seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.Affine(rotate=(-15, 15))),
                iaa.Sometimes(0.3, iaa.EdgeDetect(alpha=(0.3, 0.8))),
                iaa.Sometimes(0.5, iaa.MotionBlur(k=iap.Choice([3, 5, 7]))),
                iaa.OneOf([
                    iaa.Dropout(p=(0, 0.3), per_channel=0.5),
                    iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.09))
                ]),
                iaa.Sometimes(0.5, iaa.AllChannelsCLAHE(clip_limit=(1, 10)))
            ])
            image = seq.augment_image(image)

        # Transform to tensor
        image = TF.to_tensor(image)

        # Transform to one hot encoding
        target = torch.tensor(self.target_dict[target])

        #normalize image to fit pretrained vgg model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        image = normalize(image)

        return image, target, condition

    def make_dataset(self):
        """Loads files and creates the dataset 
        """        
        # fill samples with path and target of data
        self.samples = []
        target_classes = []
        for directory in os.listdir(self.data_path):

            file_paths = glob(os.path.join(self.data_path,directory,'*.npy'))
            target_classes.append(directory)

            for fl in file_paths:
                
                self.samples.append({"target": directory, "path": fl})

        target_classes.sort()
        self.target_dict = {k: v for v, k in enumerate(target_classes)}

    def get_target_dict(self) -> dict:
        """Returns the target dic

        Returns:
            dict: Dictionary with targets
        """        
        return self.target_dict

    def __getitem__(self, idx: int) -> dict:
        """Returns an item in the dataset

        Args:
            idx (int): Items id

        Returns:
            dict: Item as dict of "image", "target" and "condition"
        """        
        # load data
        item = np.load(self.samples[idx]["path"], allow_pickle=True)
        item = item.item()

        img = item["obs"]
        tar = self.samples[idx]["target"]
        cond = item["condition"]

        # transform
        image, target, condition = self.transform(img, tar, cond)
        
        sample = {"image": image, "target": target, "condition": condition}
        return sample
