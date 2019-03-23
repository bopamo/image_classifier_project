# Adapted from https://medium.freecodecamp.org/how-to-build-the-best-image-classifier-3c72010b3d55

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

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
        logger.debug("GPU is set")

    else:
        device = torch.device("cpu")
        logger.debug("GPU is NOT set")

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
    if device == torch.device('cpu'):
        logger.debug("Loading model with CPU")
        checkpoint = torch.load(file_path, map_location='cpu')
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

    image = process_image(image)

    # Convert 2D image to 1D vector
    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    model.eval()
    inputs = Variable(image).to(device)
    logits = model.forward(inputs)

    logger.debug(type(logits))

    logger.debug(type(image))

    ps = torch.nn.functional.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)


# Display top k predictions
def get_predictions(image_path, model, topk=5):
    """ Function for extracting a list of probabilities and
        the corresponding labels
    """
    probabilities, classes = predict(image_path, model, topk)

    labels = []

    for e in classes:
        label = "{}".format(e)

        labels.append(get_label(label))

    return probabilities, labels
