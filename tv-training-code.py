#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T
import utils
from engine import train_one_epoch, evaluate
import configs


__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017-2019"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


class PennFudanDataset:
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # Load all images files, sorting them to ensure that they are aligned.
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        # Load images and masks.
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        # note that we haven't to RGB yet.
        mask = Image.open(mask_path)
        # Convert the PIL Image to a numpy array.
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # Split the color-encoded mask into a set of binary masks.
        masks = mask == obj_ids[:, None, None]

        # Get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert everything into a torch.Tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # There is only one class.
        labels = torch.ones((num_objs), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) #todo: why not tensor?
        # Suppose all instances are not crowd.
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target


    def __len__(self):
        return len(self.imgs)



def show_landmark_roi(image, landmarks):
    """
    image and landmarks are an element from data_loader.

    :param image: type of tensor, C x W x H
    :param landmarks:
    masks: N x 1 x W x H, tensor
    :return:
    """

    # To cpu and numpy.
    image = image.detach().cpu().numpy()
    masks = landmarks['masks'].detach().cpu().numpy()
    masks = np.squeeze(masks) # N x W x H

    # Color RoI using masks.
    image = np.transpose(image, (1, 2, 0)) * 255

    for idx, mask in enumerate(masks):
        # Draw masks.
        color = np.random.choice(range(256), size=3)
        w, h = tuple(mask.shape)
        foreground_img = np.tile(color, (w, h, 1))
        mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        image = mask * foreground_img + (1-mask) * image

    image = Image.fromarray(np.uint8(image))

    draw =ImageDraw.Draw(image) # We need to draw something on the original image.

    for idx, box in enumerate(landmarks['boxes']):
        # Draw rectangles.
        x0, y0, _, _ = box
        draw.rectangle([*box], outline='green', width=3)
        draw.text((x0+1, y0+1), str(idx+1), stroke_with=7)

    image.save('detection_image.jpg')


# dataset = PennFudanDataset('../data/PennFudanPed/', None)
# show_landmark_roi(*dataset[5])


def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def main():

    device = configs.device

    # Number of class: background and person.
    num_classes = 2

    # Datasets.
    dataset = PennFudanDataset('../data/PennFudanPed/', get_transform(train=True))
    dataset_test = PennFudanDataset('../data/PennFudanPed/', get_transform(train=False))

    # Split the dataset in train and test set.
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Define training and validation data loaders.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn
    )

    # Get the model using our helper function.
    model = get_model_instance_segmentation(num_classes)

    # Move model to the right device.
    # model.to(device)

    # Construct an optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    # and a learning rate scheduler.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    # Let's train it for 10 epochs.
    num_epochs = 2#10
    for epoch in range(num_epochs):
        # Train for one epoch.
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # Update the learning rate.
        lr_scheduler.step()
        # Evaluate on the test dataset.
        evaluate(model, data_loader_test, device=device)

    print('That is all!')

    model.eval()
    for test_images, targets in data_loader_test:
        test_images = [images.to(device) for images in list(test_images)]
        output = model(test_images)

        show_landmark_roi(test_images[0], output[0])
        break


main()