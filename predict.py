#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# TODO 0: Add your information below for Programmer & Date Created.
# PROGRAMMER: Bopamo Osaisai
# DATE CREATED: Feb. 21, 2019
# REVISED DATE:
# PURPOSE: Classifies flower images using a pre-trained CNN model, compares these
#          classifications to the true identity of the flowers in the images, and
#          summarizes how well the CNN performed on the image classification task.
#          Note that the true identity of the flower (or object) in the image is
#          indicated by the filename of the image.
#
#   Example call:
#    python predict.py ./flowers/test/69/image_05959.jpg ./trained_models/densenet_model_checkpoint.pth
##

# Imports python modules

# Imports functions created for this program

from get_predict_input_args import *
from utilities_functions import *

import time


# Main program function defined below
def main():
    # STEP 0: Check Versions
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 0: Check Versions"
    log_program_step(step)
    start_time = time.time()

    logger.debug(sys.version)

    logger.debug("torch.version : {}".format(torch.version.__version__))

    # STEP 1: Get input arguments with the function get_predict_input_args()
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 1: Get input arguments with the function get_predict_input_args()"
    log_program_step(step)

    in_arg = get_predict_input_args()

    # Function that checks command line arguments using in_arg
    # check_command_line_arguments(in_arg)

    # Assign and display all arguments
    image_file = in_arg.image
    checkpoint_file = in_arg.checkpoint
    top_k = in_arg.top_k
    mapping_file = in_arg.category_names
    GPU_FLAG = in_arg.gpu

    logger.debug("image \t: {}".format(image_file))
    logger.debug("checkpoint \t: {}".format(checkpoint_file))
    logger.debug("top_k \t: {}".format(top_k))
    logger.debug("mapping \t: {}".format(mapping_file))
    logger.debug("GPU \t: {}".format(GPU_FLAG))

    device = get_device(GPU_FLAG)

    logger.debug("GPU \t: {}".format(device))

    # STEP 2: Load and verify the checkpoint
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 2: Load the checkpoint and retrieve trained model"
    log_program_step(step)

    PATH_OF_CHECKPOINT_FILE = checkpoint_file

    logger.debug("Retrieving trained model..")
    saved_model, class_to_idx, _, _ = load_checkpoint(PATH_OF_CHECKPOINT_FILE, device)

    logger.info("Trained model successfully retrieved..")

    logger.debug(type(saved_model))

    logger.debug(type(class_to_idx))

    # STEP 3: Class Prediction
    # Predict top k classes
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 3: Class Prediction"
    log_program_step(step)

    PATH_OF_IMAGE_FILE = image_file

    logger.debug("PATH_OF_IMAGE_FILE : {}".format(PATH_OF_IMAGE_FILE))

    probs, labels = get_predictions(PATH_OF_IMAGE_FILE, saved_model.to(device), top_k)

    print(probs)
    print(labels)

    # TODO 0: Measure total program runtime by collecting end time
    end_time = time.time()

    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    print(spacing_string)
    tot_time = end_time - start_time
    # print("\n** Total Elapsed Runtime:",
    #       str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":"
    #       + str(int((tot_time % 3600) % 60)))

    final_time = str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":" + str(
        int((tot_time % 3600) % 60))

    logger.info("** Total Elapsed Runtime: {}".format(final_time))


# Call to main function to run the program
if __name__ == "__main__":
    main()
