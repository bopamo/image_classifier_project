import copy
import time
from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim
from torchvision import models

from get_train_input_args import *
from get_training_data import *
from logger import *

# These are the possible models that can be trained
densenet161 = models.densenet161(pretrained=True)
vgg19 = models.vgg19(pretrained=True)

models = {'densenet': densenet161, 'vgg': vgg19}


# TODO: Build and train your network

# 2: Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

def get_model(model_name):
    if model_name == 'densenet':
        model = models[model_name]
    elif model_name == 'vgg':
        model = models[model_name]
    else:
        print("ERROR: Unknown model, please choose 'densenet' or 'vgg'")
        sys.exit(1)

    return model


def build_classifier(model, hidden_units, dropout=0.2):
    number_of_outputs = 102

    if model == models['densenet']:
        number_of_inputs = 2208
    elif model == models['vgg']:
        number_of_inputs = 25088
    else:
        print("ERROR: Unknown model, please choose 'densenet' or 'vgg'")
        sys.exit(1)

    # # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(number_of_inputs, hidden_units)),
        ('relu0', nn.ReLU()),
        ('drop0', nn.Dropout(dropout)),
        ('fc1', nn.Linear(hidden_units, number_of_outputs)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    return classifier


def configure_model(model, classifier, learn_rate=0.001, step_size=4, gamma=0.1):
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Decay LR by a factor of 0.1 every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return model, criterion, optimizer, scheduler


# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(device, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def check_model_accuracy(model, device, dataloader):
    model.eval()

    accuracy = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Class with the highest probability is our predicted class
        equality = (labels.data == outputs.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test accuracy: {:.3f}".format(accuracy/len(dataloader)))


def get_model_test():
    for model in models:
        m = get_model(model)
        print(m)

        for i in range(10):
            print('-' * 10)

    return None


def build_classifier_test():
    for model in models:
        m = get_model(model)
        test_classifier = build_classifier(m, 1000)
        print(test_classifier)

    return None


def configure_model_test():
    for model in models:
        m = get_model(model)
        cl = build_classifier(m, 1000)
        m, cr, opt, sched = configure_model(m, cl, )
        print(m)
        print(cr)
        print(opt)
        print(sched)

    return None


def train_model_test():


    logger.debug("this is a debugging message")
    logger.info("this is a debugging message")
    logger.warn("this is a debugging message")
    logger.error("this is a debugging message")
    logger.critical("this is a debugging message")

    device = get_device()

    # print(device)

    data_dir = get_data_dir()

    # print(data_dir)


    image_datasets = get_image_dataset(data_dir)

    dataloaders = get_dataloader(image_datasets)

    dataset_sizes = get_data_sizes(image_datasets)

    model_name = get_arch()

    # print(model_name)


    m = get_model(model_name)

    hidden_units = get_hidden_units()

    cl = build_classifier(m, hidden_units)
    m, cr, opt, sched = configure_model(m, cl)



    # train_model(device, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10)

    # with active_session():
    model = train_model(device, m, cr, opt, sched, dataloaders, dataset_sizes)