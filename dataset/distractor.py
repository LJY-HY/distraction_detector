from PIL import Image
import numpy as np
import torch
import csv
import os
import torchvision
from matplotlib import image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

def make_dataloader(args):
    #split subject according to args.source
    source_subject = []
    target_subject = []
    f = open('/home/esoc/datasets/driver_detection/biased_list.csv','r',encoding='utf-8-sig',newline='')
    rdr = csv.reader(f)
    for line in rdr:
        if len(args.source)==1:
            if line[1] in args.source or line[2] in args.source:
                source_subject.append(line[0])
            else:
                target_subject.append(line[0])
        elif len(args.source)==2:
            if line[1] in args.source and line[2] in args.source:
                source_subject.append(line[0])
            else:
                target_subject.append(line[0])
    f.close()

    source_img = []
    source_label = []
    target_img = []
    target_label = []

    f = open('/home/esoc/datasets/driver_detection/driver_imgs_list.csv','r',encoding='utf-8-sig',newline='')
    rdr = csv.reader(f)
    for line in rdr:
        if line[0] in source_subject:
            source_img.append(line[2])
            source_label.append(line[1])
        elif line[0] in target_subject:
            target_img.append(line[2])
            target_label.append(line[1])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    f.close()
    source_dataset = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/train/', transform = transform_train)
    target_dataset = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/train', transform = transform_test)
    s_train_dataset, s_test_dataset = random_split(source_dataset,[17424,5000])
    t_train_dataset, t_test_dataset = random_split(source_dataset,[17424,5000])

    s_train_dataloader = DataLoader(s_train_dataset, batch_size = args.batch_size, shuffle=True,num_workers=4)
    s_test_dataloader = DataLoader(s_test_dataset, batch_size = args.batch_size, shuffle=True,num_workers=4)
    t_train_dataloader = DataLoader(t_train_dataset, batch_size = args.batch_size, shuffle=True,num_workers=4)
    t_test_dataloader = DataLoader(t_test_dataset, batch_size = args.batch_size, shuffle=True,num_workers=4)

    return s_train_dataloader, s_test_dataloader, t_train_dataloader, t_test_dataloader