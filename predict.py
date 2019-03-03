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


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    # start_time = time()

    print(sys.version)

    logger.info("torch.version : {}".format(torch.version.__version__))

    # Replace sleep(75) below with code you want to time
    #     sleep(0)

    # TODO 1: Get input arguments with the function get_predict_input_args()
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg

    step = "TODO 1: Get input arguments with the function get_predict_input_args()"
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

    GPU_FLAG = False

    logger.info("image \t: {}".format(image_file))
    logger.info("checkpoint \t: {}".format(checkpoint_file))
    logger.info("top_k \t: {}".format(top_k))
    logger.info("mapping \t: {}".format(mapping_file))
    logger.info("GPU \t: {}".format(GPU_FLAG))



    device = get_device(GPU_FLAG)

    logger.info("GPU \t: {}".format(device))

    # TODO 2: Load and verify the checkpoint
    # Once the get_pet_labels function has been defined replace 'None'
    # in the function call with in_arg.dir  Once you have done the replacements
    # your function call should look like this:
    #             get_pet_labels(in_arg.dir)
    # This function creates the results dictionary that contains the results,
    # this dictionary is returned from the function call as the variable results

    step = "TODO 2: Load and verify the checkpoint"
    log_program_step(step)

    PATH_OF_CHECKPOINT_FILE = checkpoint_file

    logger.debug("Retrieving saved model..")
    saved_model, class_to_idx, _, _ = load_checkpoint(PATH_OF_CHECKPOINT_FILE, device)

    logger.debug("Saved model successfully retrieved..")

    logger.info(type(saved_model))

    logger.info(type(class_to_idx))

    # logger.info(type(saved_epochs))
    #
    # logger.info(type(saved_optimizer))

    # logger.info("NN  \t: \n{}".format(saved_model))

    # logger.info("class_to_idx  \t: \n{}".format(class_to_idx))

    # TODO 3: Process Image
    # Once the classify_images function has been defined replace first 'None'
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this:
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels,
    # and adds these results to the results dictionary - results
    # classify_images(in_arg.dir, results, in_arg.arch)

    step = "TODO 2: Load the data from the data directory"
    log_program_step(step)

    # with Image.open(image_file) as image:
    #     plt.imshow(image)

    # TODO 4: Class Prediction
    # Once the adjust_results4_isadog function has been defined replace 'None'
    # in the function call with in_arg.dogfile  Once you have done the
    # replacements your function call should look like this:
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly
    # classified images as 'a dog' or 'not a dog'. This demonstrates if
    # model can correctly classify dog images as dogs (regardless of breed)
    # adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results

    step = "TODO 4: Class Prediction"
    log_program_step(step)

    PATH_OF_IMAGE_FILE = image_file

    logger.info("PATH_OF_IMAGE_FILE : {}".format(PATH_OF_IMAGE_FILE))

    probs, labels = get_predictions(PATH_OF_IMAGE_FILE, saved_model.to(device), device, top_k)

    print(probs)
    print(labels)



    # TODO 5: Sanity Checking
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    # results_stats = calculates_results_stats(results)

    step = "TODO 5 Sanity Checking"
    log_program_step(step)


    # TODO 6: Perform test inference on saved model to make sure
    # Once the print_results function has been defined replace 'None'
    # in the function call with in_arg.arch  Once you have done the
    # replacements your function call should look like this:
    #      print_results(results, results_stats, in_arg.arch, True, True)
    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    # print_results(results, results_stats, in_arg.arch, True, True)

    # TODO 0: Measure total program runtime by collecting end time
    # end_time = time()

    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    print(spacing_string)
    tot_time = 0  # end_time - start_time
    # print("\n** Total Elapsed Runtime:",
    #       str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":"
    #       + str(int((tot_time % 3600) % 60)))

    final_time = str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":" + str(
        int((tot_time % 3600) % 60))

    logger.info("** Total Elapsed Runtime: {}".format(final_time))


# Call to main function to run the program
if __name__ == "__main__":
    main()
