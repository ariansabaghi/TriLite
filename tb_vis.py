import os
from config import *
import cv2
from torchvision import transforms
from PIL import Image

from util import t2n
import numpy as np
from inference import normalize_scoremap

import random 
import requests


# Set the seed for reproducibility
seed = 42
random.seed(seed)

# Generate 100 unique random numbers between 0 and 999
selected_classes = set(random.sample(range(1000), 100))

# Create a mapper dictionary to map the selected classes to the range 0-99
class_mapper = {new_class: original_class for new_class, original_class in enumerate(selected_classes)}

# URL to the ImageNet class index file from a popular repository
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

response = requests.get(url)
class_names = response.text.strip().split("\n")

# Create the dictionary mapping
imagenet_dict = {i: class_name for i, class_name in enumerate(class_names)}


def label_mapping(predicted_class, return_name=True):
    original_class_number = class_mapper[int(predicted_class)]
    class_name = imagenet_dict[original_class_number]
    if return_name:
        return class_name
    else:
        return original_class_number


_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def generate_overlay_heatmap(model, args, image_dir="./tb_vis_images"):

    transform = transforms.Compose([
        transforms.Resize((args.resize_eval, args.resize_eval)),
        transforms.ToTensor(),
        transforms.Normalize(args.IMAGE_MEAN_VALUE, args.IMAGE_STD_VALUE)
    ])

    overlays = []
    for file in os.listdir(image_dir):
        if args.dataset_name.lower() in file.lower():
            image_path = os.path.join(image_dir, file)
            image_orig = Image.open(image_path)
            image_size = image_orig.size
            normalized_image_orig_tensor = transforms.ToTensor()(image_orig)
            # Apply any specified transformations to the image
            image = transform(image_orig)
            images = image.to(args.device)

            with torch.no_grad():
                logits, *_, localization_map = model(images.unsqueeze(0))


            heatmap = t2n(localization_map)[0]

            heatmap = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_LINEAR)
            heatmap = normalize_scoremap(heatmap)

            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]

            overlayed = 0.5 * normalized_image_orig_tensor.permute(1, 2, 0).numpy() + 0.3 * heatmap
            overlayed = np.clip(overlayed, 0, 1)

            overlays.append(torch.tensor(overlayed.copy()).permute(2, 0, 1))


    return overlays

