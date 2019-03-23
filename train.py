#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# TODO 0: Add your information below for Programmer & Date Created.
# PROGRAMMER: Bopamo Osaisai
# DATE CREATED: Feb. 21, 2019
# REVISED DATE:
# PURPOSE: train.py will train a new network on a dataset and save the model as a checkpoint
#
##

# Imports python modules


# Imports functions created for this program

from classifier import *

from utilities_functions import *


# Main program function defined below
def main():
    # STEP 0: Check Versions

    step = "STEP 0: Check Versions"
    log_program_step(step)
    start_time = time.time()

    logger.debug(sys.version)

    logger.debug("torch.version : {}".format(torch.version.__version__))

    # STEP 1: Define get_input_args function within the file get_train_input_args.py
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 1: Define get_input_args function within the file get_train_input_args.py"
    log_program_step(step)

    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    # check_command_line_arguments(in_arg)

    # Assign and display all arguments
    data_directory = in_arg.data_dir
    save_directory = in_arg.save_dir
    model_name = in_arg.arch
    hidden_units = in_arg.hidden_units
    learning_rate = in_arg.learning_rate
    epochs = in_arg.epochs
    GPU_FLAG = in_arg.gpu

    logger.debug("data_directory : {}".format(data_directory))
    logger.debug("save_directory : {}".format(save_directory))
    logger.debug("model_name : {}".format(model_name))
    logger.debug("hidden_units : {}".format(hidden_units))
    logger.debug("learning_rate : {}".format(learning_rate))
    logger.debug("epochs : {}".format(epochs))
    logger.debug("GPU : {}".format(GPU_FLAG))

    device = get_device(GPU_FLAG)

    logger.debug(device)

    logger.debug(type(device))

    # STEP 2: Load the data from the data directory
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 2: Load the data from the data directory"
    log_program_step(step)

    image_datasets = get_image_dataset(in_arg.data_dir)

    dataloaders = get_dataloader(image_datasets)

    dataset_sizes = get_data_sizes(image_datasets)

    random_count = 10
    logger.info("Printing {} random labels and corresponding names".format(random_count))

    for num in range(random_count):
        get_random_image_and_label(dataloaders['valid'])


    # STEP 3: Define, build and train the network
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 3: Define, build and train the network"
    log_program_step(step)

    logger.info("Testing classifier...")

    logger.info("Retrieving model....")

    model_to_be_trained = get_model(model_name)

    logger.info("Model received....")

    logger.info("Building Classifier....")

    classifier = build_classifier(model_to_be_trained, hidden_units)

    logger.info("Classifier Built....")

    logger.info("Configuring model....")

    model_to_be_trained, criterion, optimizer, scheduler = configure_model(model_to_be_trained, classifier,
                                                                           learning_rate)

    logger.info("Model configured....")

    logger.info("Begin training model....")

    model_to_be_trained.to(device)

    trained_model = train_model(device, model_to_be_trained, criterion, optimizer, scheduler, dataloaders,
                                dataset_sizes, epochs)

    logger.info("Model successfully trained....\n\n")

    logger.info("Begin checking model accuracy....")

    # check_model_accuracy(trained_model, device, dataloaders['valid'])
    # check_model_accuracy(trained_model, device, dataloaders['test'])

    logger.info("Finished checking model accuracy....")

    # STEP 4: Save the checkpoint
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 4: Save the checkpoint"
    log_program_step(step)

    trained_model.class_to_idx = image_datasets['train'].class_to_idx

    if model_name == 'densenet':
        checkpoint_file_name = "densenet_model_checkpoint.pth"
        number_of_inputs = 2208
    elif model_name == 'vgg':
        checkpoint_file_name = "vgg_model_checkpoint.pth"
        number_of_inputs = 25088
    else:
        logger.error("Unknown model, please choose 'densenet' or 'vgg'")

    #
    number_of_outputs = 102

    checkpoint_to_be_saved = {'input_size': number_of_inputs,
                              'output_size': number_of_outputs,
                              'epochs': epochs,
                              'batch_size': batch_size,
                              'model': models[model_name],
                              'classifier': classifier,
                              'scheduler': scheduler,
                              'optimizer': optimizer.state_dict(),
                              'state_dict': trained_model.state_dict(),
                              'class_to_idx': trained_model.class_to_idx
                              }

    logger.debug("save_directory : {}".format(save_directory))

    PATH_TO_SAVE_CHECKPOINT = os.path.join(save_directory,
                                           checkpoint_file_name)  # inserting a list of strings os.path.join

    logger.debug("PATH_OF_CHECKPOINT : {}".format(PATH_TO_SAVE_CHECKPOINT))

    torch.save(checkpoint_to_be_saved, PATH_TO_SAVE_CHECKPOINT)

    logger.info("checkpoint_to_be_saved successfully saved...")

    saved_checkpoint = torch.load(PATH_TO_SAVE_CHECKPOINT)
    saved_checkpoint.keys()

    # STEP 5: Verify the checkpoint can be loaded
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 5: Verify the checkpoint can be loaded"
    log_program_step(step)

    # device = get_device(GPU_FLAG)

    logger.debug("Retrieving saved model..")
    loaded_model, loaded_class_to_idx, _, _ = load_checkpoint(PATH_TO_SAVE_CHECKPOINT, device)

    logger.info("Saved model successfully retrieved..")

    logger.debug(loaded_model)

    logger.debug(trained_model)

    logger.debug(loaded_class_to_idx)

    # STEP 6: Perform test inference on saved model to make sure
    #################################################################################
    #################################################################################
    #################################################################################

    step = "STEP 6: Perform test inference on saved model to make sure"
    log_program_step(step)

    PATH_OF_IMAGE_FILE = 'flowers/test/69/image_05959.jpg'

    logger.debug("PATH_OF_IMAGE_FILE : {}".format(PATH_OF_IMAGE_FILE))

    loaded_model.class_to_idx = image_datasets['train'].class_to_idx

    logger.info("Getting Model Predictions...")

    probs, labels = get_predictions(PATH_OF_IMAGE_FILE, loaded_model.to(device))

    logger.info("Model Predictions Retrieved...")

    logger.info("Printing Model Predictions: ")

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

    logger.info("**** Total Elapsed Runtime: {}".format(final_time))


# Call to main function to run the program
if __name__ == "__main__":
    main()
