#!/usr/bin/env python

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Custom package
from model.vgg16 import VGG16
from model.resnet import ResNet18
from model.resnet import ResNet50
from model.wideresnet import Wide_ResNet
from model.alexnet import alexnet, alexnet_cifar

# torchattacks toolbox
import torchattacks

def attack_loader(args, net):


    # Gradient Clamping based Attack
    if args.attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=args.eps)

    elif args.attack == "bim":
        return torchattacks.BIM(model=net, eps=args.eps, alpha=1/255)

    elif args.attack == "pgd":
        return torchattacks.PGD(model=net, eps=args.eps,
                                alpha=args.eps/args.steps*2.3, steps=args.steps, random_start=True)

    elif args.attack == "cw":
        return torchattacks.CW(model=net, c=0.1, lr=0.1, steps=args.cwsteps)

    elif args.attack == "auto":
        return torchattacks.APGD(model=net, eps=args.eps)

    elif args.attack == "fab":
        return torchattacks.FAB(model=net, eps=args.eps, n_classes=args.n_classes)

    elif args.attack == "nifgsm":
        return torchattacks.NIFGSM(model=net, eps=args.eps, alpha=2/255, steps=args.steps, decay=1.0)


def network_loader(args, mean, std):
    print('Pretrained', args.pretrained)
    print('Batchnorm', args.batchnorm)
    if args.network == "resnet18":
        print('ResNet18 Network')
        return ResNet18(args)
    elif args.network == "resnet50":
        print('ResNet50 Network')
        return ResNet50(args)
    elif args.network == "vgg16":
        print('VGG16 Network')
        return VGG16(num_classes=args.n_classes)
    elif args.network == "wide":
        print('Wide Network')
        return Wide_ResNet(28, 10, 0.3, args.n_classes)
    elif args.network == "vgg16hsic":
        print('Vgg16hsic Network')
        return VGG16_HSIC(num_classes=args.n_classes)
    elif args.network == "alex":
        print('AlexNet Network')
        if args.dataset == 'tiny':
            return alexnet()
        else:
            return alexnet_cifar()


def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3
    elif args.dataset == "cifar100":
        args.n_classes = 100
        args.img_size  = 32
        args.channel   = 3


    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.img_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return trainloader, testloader


def dataset(args, train, transform):

        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)

        if args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)

        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if train \
                                    else args.data_root + '/tiny-imagenet-200/val_classified', transform=transform)
