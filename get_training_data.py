#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# PROGRAMMER: Bopamo Osaisai
# DATE CREATED: Feb. 21, 2019
# REVISED DATE:
# PURPOSE: Create the function get_training_data that creates the transforms for
#          the training, validation, and testing sets
#          This function inputs:
#           - The data folder as data_dir within get_training_data function and
#             as in_arg.data_dir for the function call within the main function.
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main.
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##


# Imports python modules

import json
import os

import numpy as np
import torch
from torchvision import datasets, transforms

from logger import *

data_sets = ['train', 'valid', 'test']

batch_size = 64
num_workers = 0

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
}


# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create
#       with this function
#
def get_image_dataset(data_dir):
    """
    Creates an image dataset from the directory path and data transforms
    Parameters:
     data_dir - The (full) path to the folder of images that are to be
                 trained by the CNN model

    Returns:
      image_dataset - An image dataset created with ImageFolder from the data transforms
    """

    image_datasets_ = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                               data_transforms[x])
                       for x in data_sets}
    return image_datasets_


def get_dataloader(image_dataset):
    """
    Creates a dataloader object from the directory path and data transforms')
    Parameters:
     data_dir - The (full) path to the folder of images that are to be
                 trained by the CNN model

    Returns:
      dataloaders_ - A dataloaders dataset created from an image data set
    """

    dataloader_ = {data_set: torch.utils.data.DataLoader(image_dataset[data_set], batch_size=batch_size,
                                                         shuffle=True, num_workers=num_workers)
                   for data_set in data_sets}

    return dataloader_


def get_data_sizes(image_dataset):
    """
    Creates a dict of sizes for the elements in the dataset
    Parameters:
     image_dataset - An image dataset created with ImageFolder from the data transforms


    Returns:
      data_set_sizes_ - A dict of sizes for each element in the image dataset
    """

    data_set_sizes_ = {data_set: len(image_dataset[data_set]) for data_set in data_sets}

    return data_set_sizes_


def create_labels():
    """
    Creates a dict of sizes for the elements in the dataset
    Parameters:
     N/A

    Returns:
      cat_to_names_ - A dict of labels mapping index values to flower names from the image dataset
    """

    with open('cat_to_name.json', 'r') as f:
        cat_to_names_ = json.load(f)

    return cat_to_names_


def get_label(index):
    """
    Creates a dict of sizes for the elements in the dataset
    Parameters:
     index - an integer that represents the index of a flower name

    Returns:
      label - A string mapped from the input index parameter
    """

    cat_to_name = create_labels()

    index_str = "{}".format(index)

    label = cat_to_name.get(index_str)

    return label


# # test data loader with random data
def get_random_image_and_label(data_loader):
    cat_to_name = create_labels()
    images, labels = next(iter(data_loader))
    rand_idx = np.random.randint(len(images))
    # print(rand_idx)
    label = "{}".format(rand_idx)
    logger.info("label: {},\tname: {}".format(labels[rand_idx].item(),
                                        cat_to_name.get(label)))
    return images, labels
