# Adapted from https://medium.freecodecamp.org/how-to-build-the-best-image-classifier-3c72010b3d55

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import helper
from get_training_data import get_label
from logger import *

GPU_FLAG = False


def print_cute(value, count):
    print("\n")

    for i in range(2):
        print(value * count)

    print("\n")


def get_device(gpu_is_enabled):
    if gpu_is_enabled is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("GPU is set")

    else:
        device = torch.device("cpu")
        logger.info("GPU is NOT set")

    # print(device)

    train_on_gpu = torch.cuda.is_available()

    # print_cute('*', 20)
    if not train_on_gpu:
        logger.info('Bummer!  Training on CPU ...')
    else:
        logger.info('You are good to go!  Training on GPU ...')
    # print_cute('*', 20)

    return device


def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(device)

    train_on_gpu = torch.cuda.is_available()

    # print_cute('*', 20)
    if not train_on_gpu:
        logger.info('Bummer!  Training on CPU ...')
    else:
        logger.info('You are good to go!  Training on GPU ...')
    # print_cute('*', 20)

    return device


# Image utility functions

def load_checkpoint(file_path, device):
    if device is 'cpu':
        logger.debug("Loading model with CPU")
        checkpoint = torch.load(file_path, map_location=device)
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    else:
        logger.debug("Loading model with GPU!!!!")
        checkpoint = torch.load(file_path)
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        model.to(device)

    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint['class_to_idx'], epochs, optimizer


# def imshow(image, ax=None, title=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.transpose((1, 2, 0))
#
#     # Undo pre-processing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#
#     return ax


# def show_images(data_loader, count, cat_to_name):
#     data_iter = iter(data_loader)
#
#     images, labels = next(data_iter)
#
#     fig, axes = plt.subplots(figsize=(10, 4), ncols=count)
#
#     for ii in range(count):
#         ax = axes[ii]
#         title = "{}".format(labels[ii].item() + 1)
#         helper.imshow(images[ii], ax=ax, title=cat_to_name.get(title))


# show_images(dataloaders['train'], 4)
# show_images(dataloaders['valid'], 4)
# show_images(dataloaders['test'],  4)

def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # Process a PIL image for use in a PyTorch model
    #     tensor.numpy().transpose(1, 2, 0)
    pre_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = pre_process(image)
    image = np.array(image)
    return image



def predict(image_path, model, topk=5):
    """
        Predict the class (or classes) of an image using a trained deep learning model.
    """
    device = get_device(GPU_FLAG)

    # Implement the code to predict the class from an image file
    image = Image.open(image_path)

    logger.info(type(image))

    image = process_image(image)

    # Convert 2D image to 1D vector
    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    logger.info(type(image))

    model.eval()
    inputs = Variable(image).to(device)
    logits = model.forward(inputs)

    logger.info(type(logits))

    logger.info(type(image))

    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)


def get_predictions(image_path, model, topk=5):
    """ Function for extracting a list of probabilities and
        the corresponding labels
    """
    probabilities, classes = predict(image_path, model, topk)

    labels = []

    #     print(probs)
    #     print(classes)
    for e in classes:
        label = "{}".format(e)
        #         print(name)
        labels.append(get_label(label))

    return probabilities, labels


# TODO: Display an image along with the top 5 classes

# def display_image_predictions(image_path, probabilities, labels):
#     """ Function for viewing an image and it's predicted classes.
#     """
#     image = Image.open(image_path)
#
#     fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), ncols=1, nrows=2)
#     ax1.set_title(labels[0])
#     ax1.imshow(image)
#     ax1.axis('off')
#
#     y_pos = np.arange(len(probabilities))
#     ax2.barh(y_pos, probabilities, align='center')
#     ax2.set_yticks(y_pos)
#     ax2.set_yticklabels(labels)
#     ax2.invert_yaxis()  # labels read top-to-bottom
#     ax2.set_title('Class Probability')

