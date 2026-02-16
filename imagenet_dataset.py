import os.path as osp
import lmdb
import pickle
import time
import torch
import io
import sys

from PIL import Image
from torchvision.io import decode_jpeg, ImageReadMode
from torchvision import transforms
from torch.utils.data import DataLoader


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, args, backend='Pillow', mode='train', resize_size=256, crop_size=224):
        """
            Initialize the Dataset object.
            
            Keyword Arguments:
                db_path : the location of the '.lmdb' object as a path-like (f.e. str, os.path).
                transform : the transformation done on each image, either None or a Callable that has a Torch.Tensor as input and as output.
                target_transform : the transformation done on each target / label, either None or a Callable that has a Torch.Tensor as input and as output.
                backend : either 'Pillow' or 'Vision'. The Pillow backend is also used by the ImageFolder class (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html),
                                                       and as such the validation accuracy of the pretrained model is exactly the same as with the ImageFolder class. 
                                                       Vision is the TorchVision implementation which uses the nvjpeg library, which outputs a Tensor and can also be GPU-accelerated. 
                                                       However, the resulting JPEGs are slightly different from those decoded by Pillow. 
                                                       This results in a slightly lower accuracy on a pre-trained ResNet-50 model, (80.846% vs 79.826%), however we 
                                                       hypothesise that training the model from scratch with images decoded by nvjpeg will reach similar accuracies. 
                                                       !!! Right now, we recommend using Pillow as backend, because it is only slightly slower, but performs better with pretrained models. !!!
        """
        
        self.mode = mode
        # parent_dir = "/media/arian/Extreme SSD/imagenet"
        parent_dir = "/mnt/windows/datasets/imagenet"
        # parent_dir = "/project_antwerp/wsol_datasets/imagenet_100"

        print(f"Using file path: {parent_dir}")

        # if mode == "train":
        #     db_path = f"{parent_dir}/train/100_train.lmdb"
        # elif mode == "val":
        #     db_path = f"{parent_dir}/val/100_val2.lmdb"
        # elif mode == "test":
        #     db_path = f"{parent_dir}/test/100_val.lmdb"

        if mode == "train":
            db_path = f"{parent_dir}/train.lmdb"
        elif mode == "val":
            db_path = f"{parent_dir}/val.lmdb"
        elif mode == "test":
            db_path = f"{parent_dir}/test.lmdb"

        self.environment = lmdb.open(db_path, subdir=osp.isdir(db_path),
                                readonly=True, lock=False,
                                readahead=False, meminit=False, 
                                metasync=False, sync=False)
        
        with self.environment.begin(write=False) as txn:
            try:
                self.length = pickle.loads(txn.get(b'__len__'))
            except:
                self.length = int(txn.get(b'__len__').decode('ascii'))


        self.backend = backend
        self.transforms = dict(
            train=transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.IMAGE_MEAN_VALUE, args.IMAGE_STD_VALUE)
            ]),
            val=transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(args.IMAGE_MEAN_VALUE, args.IMAGE_STD_VALUE)
            ]),
            test=transforms.Compose([
                transforms.Resize((args.resize_eval, args.resize_eval)),
                transforms.ToTensor(),
                transforms.Normalize(args.IMAGE_MEAN_VALUE, args.IMAGE_STD_VALUE)
            ]))

    def __getitem__(self, index):
        """
            Implements the map-style iteration of the Dataset, as specified in https://pytorch.org/docs/stable/data.html#map-style-datasets.
            
            Returns a (image, label) tuple of two torch.Tensors
        """        
        if index >= len(self): raise IndexError  # This will allow us to use iter(dataset), as well as some additional safety against user actions.
                
            
        with self.environment.begin(write=False) as txn:
            byteflow = txn.get(u'{}'.format(index).encode('ascii'))
        unpacked = pickle.loads(byteflow)        
        
        if self.backend == 'Vision':      
            imgbuf = bytearray(unpacked[0])  # Convert immutable (non-writeable) bytes object to mutable (writeable) bytearray object.
            imgbuf = torch.frombuffer(imgbuf, dtype=torch.uint8)

            img = decode_jpeg(imgbuf, mode=ImageReadMode.RGB, device='cpu')  # Currently only on CPU, because GPU decoding is in beta and requires CUDA >= 11.6: https://pytorch.org/vision/stable/generated/torchvision.io.decode_jpeg.html
            img = img.float() / 255

        elif self.backend == 'Pillow':
            img = Image.open(io.BytesIO(unpacked[0])).convert("RGB")
        
        else:
            raise ValueError(f"Backend {self.backend} is invalid. Either use 'Pillow' or 'Vision', the recommended option is 'Pillow'.")
        
        # load label
        target = unpacked[1]

        # Apply any specified transformations to the image
        img = self.transforms[self.mode](img)
        if self.mode == "train":
            return img, torch.tensor(target), ''
        elif self.mode == "val":
            return img, torch.tensor(target), "val2/" + unpacked[2]
        elif self.mode == "test":
            return img, torch.tensor(target), "val/" + unpacked[2] + '.JPEG'
        else:
            raise ValueError(f"mode {self.backend} is invalid.")
        
        return img, torch.tensor(target), unpacked[2]

    def __len__(self):
        """
            Implements the map-style iteration of the Dataset, as specified in https://pytorch.org/docs/stable/data.html#map-style-datasets.
        """
        return self.length

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.db_path}) : containing {len(self)} samples.'


if __name__ == "__main__":

    dataset = LMDBDataset(mode="val")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    counter = 10
    for img, target, image_id in dataloader:
        print(image_id)
        counter -= 1
        if counter == 0:
            break
