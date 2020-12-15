from PIL import Image
import numpy as np
import torch
import csv
import os
import torchvision
import shutil
from matplotlib import image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

def make_dataloader(args):    
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    source_dataset = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/train/', transform = transform_train)
    s_train_dataset, s_test_dataset = random_split(source_dataset,[17424,5000])

    s_train_dataloader = DataLoader(s_train_dataset, batch_size = args.batch_size, shuffle=True,num_workers=4)
    s_test_dataloader = DataLoader(s_test_dataset, batch_size = args.batch_size, shuffle=True,num_workers=4)

    return s_train_dataloader, s_test_dataloader

def make_biased_dataloader(args):
    #split subject according to args.source
    source_subject_train = []
    source_subject_test = []
    target_subject_train = []
    target_subject_test = []
    f = open('/home/esoc/datasets/driver_detection/biased_list.csv','r',encoding='utf-8-sig',newline='')
    rdr = csv.reader(f)
    for line in rdr:
        if len(args.source)==1:
            if line[1] in args.source or line[2] in args.source:
                source_subject_train.append(line[0])
            else:
                target_subject_train.append(line[0])
        elif len(args.source)==2:
            if line[1] in args.source and line[2] in args.source:
                source_subject_train.append(line[0])
            else:
                target_subject_train.append(line[0])
    f.close()
    if 'subject' in target_subject_train:
        target_subject_train.remove('subject')

    source_subject_test.append(source_subject_train.pop(np.random.randint(len(source_subject_train))))
    target_subject_test.append(target_subject_train.pop(np.random.randint(len(target_subject_train))))

    source_img_train = []
    source_label_train = []
    target_img_train = []
    target_label_train = []

    source_img_test = []
    source_label_test = []
    target_img_test = []
    target_label_test = []

    f = open('/home/esoc/datasets/driver_detection/driver_imgs_list.csv','r',encoding='utf-8-sig',newline='')
    rdr = csv.reader(f)
    for line in rdr:
        if line[0] in source_subject_train:
            source_img_train.append(line[2])
            source_label_train.append(line[1])
        elif line[0] in source_subject_test:
            source_img_test.append(line[2])
            source_label_test.append(line[1])
        elif line[0] in target_subject_train:
            target_img_train.append(line[2])
            target_label_train.append(line[1])
        elif line[0] in target_subject_test:
            target_img_test.append(line[2])
            target_label_test.append(line[1])    
    f.close()

    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    source_imsave(source_img_train,source_label_train,source_img_test,source_label_test)
    target_imsave(target_img_train,target_label_train,target_img_test,target_label_test)
    source_dataset_train = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/biased_train/source/train', transform = transform_train)
    source_dataset_test = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/biased_train/source/test', transform = transform_test)
    target_dataset_train = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/biased_train/target/train', transform = transform_train)
    target_dataset_test = ImageFolder(root = '/home/esoc/datasets/driver_detection/imgs/biased_train/target/test', transform = transform_test)

    source_train_dataloader = DataLoader(source_dataset_train, batch_size = args.batch_size, shuffle=True,num_workers=4)
    source_test_dataloader = DataLoader(source_dataset_test, batch_size = args.batch_size, shuffle=True,num_workers=4)
    target_train_dataloader = DataLoader(target_dataset_train, batch_size = args.batch_size, shuffle=True,num_workers=4)
    target_test_dataloader = DataLoader(target_dataset_test, batch_size = args.batch_size, shuffle=True,num_workers=4)

    return source_train_dataloader, source_test_dataloader, target_train_dataloader, target_test_dataloader

def source_imsave(source_img_train,source_label_train,source_img_test,source_label_test):
    path = '/home/esoc/datasets/driver_detection/imgs/biased_train/'
    # if already dir and data is in the tree, empty it and re-fill
    if os.path.exists(path+'source/'):
        shutil.rmtree(path+'source/')
    if os.path.exists(path+'target/'):
        shutil.rmtree(path+'target/')

    for count in range(len(source_img_train)):
        if not os.path.exists(path+'source/train/'+source_label_train[count]):
            os.makedirs(path+'source/train/'+source_label_train[count])
        shutil.copy(path+source_img_train[count], path+'source/train/'+source_label_train[count])
    
    for count in range(len(source_img_test)):
        if not os.path.exists(path+'source/test/'+source_label_test[count]):
            os.makedirs(path+'source/test/'+source_label_test[count])
        shutil.copy(path+source_img_test[count], path+'source/test/'+source_label_test[count])

def target_imsave(target_img_train,target_label_train,target_img_test,target_label_test):
    path = '/home/esoc/datasets/driver_detection/imgs/biased_train/'
    for count in range(len(target_img_train)):
        if not os.path.exists(path+'target/train/'+target_label_train[count]):
            os.makedirs(path+'target/train/'+target_label_train[count])
        shutil.copy(path+target_img_train[count], path+'target/train/'+target_label_train[count])
    
    for count in range(len(target_img_test)):
        if not os.path.exists(path+'target/test/'+target_label_test[count]):
            os.makedirs(path+'target/test/'+target_label_test[count])
        shutil.copy(path+target_img_test[count], path+'target/test/'+target_label_test[count])
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)