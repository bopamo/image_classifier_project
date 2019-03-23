#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#                                                                             
# PROGRAMMER: Bopamo Osaisai
# DATE CREATED: Feb. 21, 2019
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following 3 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse
import sys
import os

import torch

from logger import *


def print_cute(value, count):
    print("\n")

    for i in range(2):
        print(value * count)

    print("\n")

    return None


def checkpoint_file(value):
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(
            'argument filename does not exist')
    if not value.endswith('.pth'):
        raise argparse.ArgumentTypeError(
            'argument filename must be of type "*.pth"')
    return value


def image_file(value):
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(
            'argument filename does not exist')
    if not value.endswith('.jpg'):
        raise argparse.ArgumentTypeError(
            'argument filename must be of type "*.jpg"')
    return value


def json_file(value):
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(
            'argument filename does not exist')
    if not value.endswith('.json'):
        raise argparse.ArgumentTypeError(
            'argument filename must be of type "*.json"')
    return value


def top_k_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("top_k must be at least 1")
    if x > 10:
        raise argparse.ArgumentTypeError("top_k must be 10 or less")
    return x


# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_predict_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 4 command line arguments.
    If the user fails to provide the first argument, data_dir, the program
    should fail and exit. If the user fails to provide some or all of the other 3 arguments,
    then the default values are used for the missing arguments.
    Command Line Arguments:
      1. Image file (and corresponding path) as image
      2. Checkpoint (and corresponding path) as checkpoint

    Optional Command Line Arguments:
      3. Top KK most likely classes as --top_k with default value 5
      4. Mapping of categories to real names as --category_names with default value 'cat_to_name.json'
      5. Use GPU for inference as --gpu with default value True

    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object


    Predict flower name from an image with predict.py along with the probability of that name.
    That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

    Basic usage: python predict.py /path/to/image /path/to/checkpoint

   python3 predict.py ./flowers/test/69/image_05959.jpg ./trained_models/densenet_model_checkpoint.pth

   ./trained_models/densenet_model_checkpoint.pth

    Options:
    1. Return top KK most likely classes:
                python predict.py input checkpoint --top_k 3

    2. Use a mapping of categories to real names:
                python predict.py input checkpoint --category_names cat_to_name.json

    3. Use GPU for inference:
                python predict.py input checkpoint --gpu

    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create required data_directory argument
    # Argument 1: that's a path to an image
    parser.add_argument('image', type=image_file,
                        help='An image file of type "*.jpg"')

    # Argument 2: that's a path to an checkpoint
    parser.add_argument('checkpoint', type=checkpoint_file,
                        help='A checkpoint of type "*.pth"')

    # Create 3 optional command line arguments as mentioned above using add_argument() from ArguementParser method

    # Argument 3: that's a top KK of most likely classes
    parser.add_argument('-t', '--top_k', type=top_k_type, default=5,
                        help="An integer between  1 and 10, representing the top k most likely classes to be displayed")

    # Argument 4: that's the mapping of categories to real names via JSON file
    parser.add_argument('-json', '--category_names', type=json_file, default='cat_to_name.json',
                        help='A JSON file with index to string mappings of type "*.json"')

    # Argument 5: that's GPU
    parser.add_argument('-g', '--gpu', action="store_true", default=True,
                        help='Flag for enabling GPU')

    args = parser.parse_args()

    # Validate image path and file
    logger.debug("Validating image path and file...")
    validate_path_and_file(args.image, '.jpg')

    # Validate checkpoint path and file
    logger.debug("Validating checkpoint path and file...")
    validate_path_and_file(args.checkpoint, '.pth')

    # Validate category_names path and file
    logger.debug("Validating category_names path and file...")
    validate_path_and_file(args.category_names, ".json")

    logger.debug("All paths and files successfully validated...")

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def validate_path_and_file(directory_path, file_extension):
    if os.path.isfile(directory_path) and directory_path.endswith(file_extension):
        logger.debug("File Valid: {}".format(directory_path))
        return None

    logger.error("invalid path...")
    sys.exit(1)
    return None  # this line is unreachable


# Functions below defined to help with "Checking your code", specifically
# running these functions with the appropriate input arguments within the
# main() funtion will print out what's needed for "Checking your code"
#
def check_command_line_arguments(in_arg):
    """
    For Lab: Classifying Images - 7. Command Line Arguments
    Prints each of the command line arguments passed in as parameter in_arg,
    assumes you defined all three command line arguments as outlined in
    '7. Command Line Arguments'
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console
    """
    if in_arg is None:
        logger.info("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        logger.info("Command Line Arguments:\n")

        logger.info("image =", in_arg.image)
        logger.info("checkpoint =", in_arg.checkpoint)
        logger.info("top_k =", in_arg.top_k)
        logger.info("category_names =", in_arg.category_names)
        logger.info("GPU =", in_arg.gpu)

    return None


def get_device_arg():
    in_arg = get_predict_input_args()

    if in_arg.gpu is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.debug("GPU is set")

    else:
        device = torch.device("cpu")
        logger.debug("GPU is NOT set")

    return device
