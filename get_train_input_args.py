#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_train_input_args.py
#                                                                             
# PROGRAMMER: Bopamo Osaisai
# DATE CREATED: Feb. 21, 2019
# REVISED DATE: 
# PURPOSE: Retrieves and parses the 7 command line arguments provided by the user when
#     they run the program from a terminal window. This function uses Python's
#     argparse module to created and defined these 4 command line arguments.
#     If the user fails to provide the first argument, data_dir, the program
#     should fail and exit. If the user fails to provide some or all of the other 3 arguments,
#     then the default values are used for the missing arguments.
#     Command Line Arguments:
#       1. Data Directory as data_dir
#       2. Save Directory as --save_dir with default value of '/trained_models'
#       3. CNN Model Architecture as --arch with default value of 'vgg'
#       4. Number of Hidden Units as --hidden_units with default value of 1000
#       5. The Learning Rate as --learning_rate with default value of 0.001
#       6. Epochs as --epochs with default value of 25
#       7. GPU flags --gpu with default value of True
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


def lr_type(x):
    x = float(x)
    if x < 0.00001:
        raise argparse.ArgumentTypeError("learning rate must be at least 0.00001")
    if x > 1.0:
        raise argparse.ArgumentTypeError("learning rate must be less than 1.0")
    return x


def hidden_units_type(x):
    x = int(x)
    if x < 200:
        raise argparse.ArgumentTypeError("hidden units must be at least 200")
    if x > 2000:
        raise argparse.ArgumentTypeError("hidden units must not be greater than 2000")
    return x


def epochs_type(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError("epochs count must be at least 0")
    if x > 50:
        raise argparse.ArgumentTypeError("epochs count must not be greater than 50")
    return x


def dir_type(value):
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            'argument directory does not exist')
    return value


# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 4 command line arguments.
    If the user fails to provide the first argument, data_dir, the program
    should fail and exit. If the user fails to provide some or all of the other 3 arguments,
    then the default values are used for the missing arguments.
    Command Line Arguments:
      1. Data Directory as data_dir
      2. Save Directory as --save_dir with default value of '/trained_models'
      3. CNN Model Architecture as --arch with default value of 'vgg'
      4. Number of Hidden Units as --hidden_units with default value of 1000
      5. The Learning Rate as --learning_rate with default value of 0.001
      6. Epochs as --epochs with default value of 25
      7. GPU flags --gpu with default value of True
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create required data_directory argument
    # Argument 1: that's a path to a folder
    parser.add_argument('data_dir', type=dir_type,
                        help='Path to a data directory')

    # Create 3 optional command line arguments as mentioned above using add_argument() from ArguementParser method

    # Argument 2: that's a path to a folder
    parser.add_argument('--save_dir', type=dir_type, default='./trained_models/',
                        help="Path to which the trained model should be saved")

    # Argument 3: that's the CNN Model Architecture
    parser.add_argument('-a', '--arch', type=str, default='densenet', choices=['densenet', 'vgg'],
                        help='CNN Model Architecture')

    # Argument 4: that's the model's hidden units count
    parser.add_argument('-hu', '--hidden_units', type=hidden_units_type, default=1000,
                        help='Model Hidden Layers')

    # Argument 5: that's learning rate
    parser.add_argument('-lr', '--learning_rate', type=lr_type, default=0.001,
                        help='Rate at which the model is trained')

    # Argument 6: that's epochs
    parser.add_argument('-e', '--epochs', type=epochs_type, default=25,
                        help='Number of cycles at which the model is trained')

    # Argument 7: that's GPU
    parser.add_argument('-g', '--gpu', action="store_true", default=True,
                        help='Flag for enabling GPU')

    args = parser.parse_args()

    # Validate data directory
    logger.debug("Validating data dir...")
    validate_path(args.data_dir)

    # Validate save directory
    logger.debug("Validating save dir...")
    validate_path(args.save_dir)

    logger.debug("Directories are valid...")

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def validate_path(directory_path):
    if os.path.exists(directory_path):
        logger.debug("DIR: {}".format(directory_path))
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
        logger.info("data_dir =", in_arg.data_dir)
        logger.info("save_dir =", in_arg.save_dir)
        logger.info("arch =", in_arg.arch)
        logger.info("hidden_units =", in_arg.hidden_units)
        logger.info("learning_rate =", in_arg.learning_rate)
        logger.info("epochs =", in_arg.epochs)
        logger.info("GPU =", in_arg.gpu)

    return None


def get_device_arg():
    in_arg = get_input_args()

    if in_arg.gpu is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.debug("GPU is set")

    else:
        device = torch.device("cpu")
        logger.debug("GPU is NOT set")

    logger.debug(device)

    return device


def get_data_dir():
    in_arg = get_input_args()

    return in_arg.data_dir


def get_save_dir():
    in_arg = get_input_args()

    return in_arg.save_dir


def get_arch():
    in_arg = get_input_args()

    return in_arg.arch


def get_learning_rate():
    in_arg = get_input_args()

    return in_arg.learning_rate


def get_epochs():
    in_arg = get_input_args()

    return in_arg.epochs


def get_hidden_units():
    in_arg = get_input_args()

    return in_arg.hidden_units
