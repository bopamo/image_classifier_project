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

from logger import *

from utilities_functions import *


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    # start_time = time()

    print(sys.version)

    # logging.warning('Watch out!')  # will print a message to the console
    # logger.debug("this is a debugging message")

    # Replace sleep(75) below with code you want to time
    #     sleep(0)

    # check_gpu()

    # TODO 1: Define get_input_args function within the file get_train_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
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

    logger.info("data_directory : {}".format(data_directory))
    logger.info("save_directory : {}".format(save_directory))
    logger.info("model_name : {}".format(model_name))
    logger.info("hidden_units : {}".format(hidden_units))
    logger.info("learning_rate : {}".format(learning_rate))
    logger.info("epochs : {}".format(epochs))
    logger.info("GPU : {}".format(GPU_FLAG))

    device = get_device(GPU_FLAG)

    logger.info(device)

    logger.info(type(device))


    # TODO 2: Load the data from the data directory
    # Once the get_pet_labels function has been defined replace 'None'
    # in the function call with in_arg.dir  Once you have done the replacements
    # your function call should look like this:
    #             get_pet_labels(in_arg.dir)
    # This function creates the results dictionary that contains the results,
    # this dictionary is returned from the function call as the variable results
    image_datasets = get_image_dataset(in_arg.data_dir)

    dataloaders = get_dataloader(image_datasets)

    dataset_sizes = get_data_sizes(image_datasets)

    class_names = image_datasets['train'].classes

    logger.info(type(dataloaders))
    logger.info(type(dataset_sizes))
    logger.info(type(class_names))

    # for num in range(10):
    #     get_random_image_and_label(dataloaders['valid'])

    logger.info(get_label(23))

    # Function that checks Pet Images in the results Dictionary using results
    # check_creating_pet_image_labels(results)

    # TODO 3: Define, build and train the network
    # Once the classify_images function has been defined replace first 'None'
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this:
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels,
    # and adds these results to the results dictionary - results
    # classify_images(in_arg.dir, results, in_arg.arch)

    logger.info("Testing classifier...")

    logger.info("Retrieving model....")

    model_to_be_trained = get_model(model_name)

    logger.info("Model received....")

    logger.info("Building Classifier....")

    classifier = build_classifier(model_to_be_trained, hidden_units)

    logger.info("Classifier Built....")

    logger.debug("Configuring model....")

    model_to_be_trained, criterion, optimizer, scheduler = configure_model(model_to_be_trained, classifier,
                                                                           learning_rate)

    logger.info("Model configured....")

    logger.info("Begin training model....")

    model_to_be_trained.to(device)

    # print("The state dict keys: \n\n", model_to_be_trained.state_dict().keys())

    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in classifier.state_dict():
    #     print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())
    #
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    trained_model = train_model(device, model_to_be_trained, criterion, optimizer, scheduler, dataloaders,
                                dataset_sizes, epochs)

    logger.info(type(trained_model))

    logger.info("Model successfully trained....\n\n")

    logger.info("Begin checking model accuracy....")

    # check_model_accuracy(trained_model, device, dataloaders['valid'])
    # check_model_accuracy(trained_model, device, dataloaders['test'])

    logger.info("Finished checking model accuracy....")

    # python3 train.py ./flowers/ --save_dir ./trained_models/ --gpu

    # python train.py ./flowers/ --gpu --arch vgg -e 10 -lr 0.001

    # Function that checks Results Dictionary using results
    # check_classifying_images(results)

    # TODO 4: Save the checkpoint
    # Once the adjust_results4_isadog function has been defined replace 'None'
    # in the function call with in_arg.dogfile  Once you have done the
    # replacements your function call should look like this:
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly
    # classified images as 'a dog' or 'not a dog'. This demonstrates if
    # model can correctly classify dog images as dogs (regardless of breed)
    # adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results
    trained_model.class_to_idx = image_datasets['train'].class_to_idx

    # logger.info("save_directory : {}".format(save_directory))

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



    logger.info("save_directory : {}".format(save_directory))

    PATH_TO_SAVE_CHECKPOINT = os.path.join(save_directory, checkpoint_file_name)  # inserting a list of strings os.path.join

    logger.info("PATH_OF_CHECKPOINT : {}".format(PATH_TO_SAVE_CHECKPOINT))

    torch.save(checkpoint_to_be_saved, PATH_TO_SAVE_CHECKPOINT)

    # Loading the checkpoint
    # with active_session():
    #     saved_checkpoint = torch.load(PATH)
    #     saved_checkpoint.keys()

    saved_checkpoint = torch.load(PATH_TO_SAVE_CHECKPOINT)
    saved_checkpoint.keys()

    # TODO 5: Verify the checkpoint can be loaded
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    # results_stats = calculates_results_stats(results)

    device = get_device(GPU_FLAG)

    logger.debug("Retrieving saved model..")
    loaded_model, loaded_class_to_idx, _, _ = load_checkpoint(PATH_TO_SAVE_CHECKPOINT, device)

    logger.debug("Saved model successfully retrieved..")

    logger.debug("Verify loaded model is the same as saved model...")

    if loaded_model.classifier == model_to_be_trained.classifier:
        logger.info("Models are the same...")
    else:
        logger.warning("Models are NOT the same!!!")

    if loaded_class_to_idx == trained_model.class_to_idx:
        logger.info("Models are the same...")
    else:
        logger.warning("Models are NOT the same!!!")

    logger.info(type(loaded_model))

    logger.info(type(trained_model))

    logger.info(type(loaded_class_to_idx))

    # logger.info(type(loaded_epochs))
    #
    # logger.info(type(loaded_optimizer))

    # TODO 6: Perform test inference on saved model to make sure
    # Once the print_results function has been defined replace 'None'
    # in the function call with in_arg.arch  Once you have done the
    # replacements your function call should look like this:
    #      print_results(results, results_stats, in_arg.arch, True, True)
    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    # print_results(results, results_stats, in_arg.arch, True, True)

    PATH_OF_IMAGE_FILE = 'flowers/test/69/image_05959.jpg'

    logger.info("PATH_OF_IMAGE_FILE : {}".format(PATH_OF_IMAGE_FILE))

    device = get_device(GPU_FLAG)

    loaded_model.class_to_idx = image_datasets['train'].class_to_idx

    probs, labels = get_predictions(PATH_OF_IMAGE_FILE, loaded_model.to(device))

    print(probs)
    print(labels)

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
