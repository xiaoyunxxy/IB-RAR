#!/usr/bin/env python

# numpy package
import numpy as np

# torch package
import torch
import torchvision
from torch.nn.functional import cross_entropy
import torch.nn.functional as F

# basic package
import os
import argparse
from tqdm import tqdm
from datetime import datetime

# custom package
from loader.loader import dataset_loader, network_loader, attack_loader
from loader.argument_print import argument_print
from hsic import hsic_normalized_cca
from misc import *


# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# argument parser
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--network', default='vgg16', type=str, help='network name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='../data', type=str, help='path to dataset')
parser.add_argument('--epoch', default=60, type=int, help='epoch number')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--save_dir', default='/data/xuxx/experiment_MI/', type=str, help='save directory')
parser.add_argument('--mi_loss', default=False, type=str2bool, help='use mi loss or not')
parser.add_argument('--lx', default=0.008, type=float, help='regular for I(X,T)')
parser.add_argument('--ly', default=0.08, type=float, help='regular for I(Y,T)')
parser.add_argument('--save_IP', default='./infofiles/mi_fcfilter.npy', type=str, help='info plane save directory')
parser.add_argument('--save_LL', default='./infofiles/mi_ll_fcfilter.npy', type=str, help='info between layers save directory')
parser.add_argument('--save_fc', default='./infofiles/mi_fc_fcfilter.npy', type=str, help='info for feature channels save directory')
parser.add_argument('--info_plane', default=False, type=str2bool, help='compute IP')
parser.add_argument('--adv_plane', default=False, type=str2bool, help='compute IP of adv examples')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--fc', default=False, type=str2bool, help='feature channels')
parser.add_argument('--advinput', default=False, type=str2bool, help='use adv input to compute MI for loss')
parser.add_argument('--selec_layer', default='all', type=str, help='input 1,2,3 etc. Selecting layers for mutual information')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

# loading dataset, network

trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
if len(args.gpu_id.split(','))!=1:
    net = torch.nn.DataParallel(net)
args.eps = 0

# Adam Optimizer with KL divergence, and Scheduling Learning rate
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
criterion_kl = torch.nn.KLDivLoss(reduction='none')

# Setting checkpoint date time
date_time = datetime.today().strftime("%m%d%H%M")

# checkpoint_name
checkpoint_name = 'MILossFC_'+args.network+'_'+args.dataset+'_'+date_time+'.pth'

# argument print
argument_print(args, checkpoint_name)


# attack
args.eps = 0.03
args.steps = 10
attack = attack_loader(args, net)

# computing mutual information
num_vgg_t = 7
num_vgg_fc = 512
id_last_cov = 4
n_iterations = (50000 // (args.batch_size*50))*args.epoch
info = np.empty([n_iterations, 2, num_vgg_t])
info_fc = np.empty([n_iterations, 2, num_vgg_fc])
info_tt = np.empty([n_iterations, 6])

num_test = 200
info_x = torch.empty([num_test, 3, 32, 32])
info_y = torch.zeros([num_test, 10])
info_x = info_x.cuda()
info_y = info_y.cuda()

def train():
    current_iteration = 0

    for epoch in range(args.epoch):

        # train environment

        print('\n\n[MILossFC/Epoch] : {}'.format(epoch+1))

        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            net.train()
            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.cuda(), targets.cuda()
            # learning network parameters
            optimizer.zero_grad()
            net.record = True
            if args.fc and batch_idx % 10 == 0 and batch_idx != 0:
                net.targets = torch.nn.functional.one_hot(targets)
            outputs, intermediates = net(inputs)

            if args.selec_layer != 'all':
                intermediates = layer_selection(intermediates, args.selec_layer)

            loss = criterion(outputs, targets)
            if args.mi_loss:
                total_loss = loss
                h_target = torch.nn.functional.one_hot(targets)
                h_data = inputs.view(inputs.shape[0], -1)
                for t_index in range(len(intermediates)):
                    intermediates[t_index] = intermediates[t_index].view(intermediates[t_index].shape[0], -1)

                    hx_l, hy_l = hsic_objective(
                        intermediates[t_index],
                        h_target=h_target.float(),
                        h_data=h_data,
                        sigma=5,
                        k_type_y='linear'
                        )

                    temp_hsic = args.lx * hx_l - args.ly * hy_l
                    total_loss += temp_hsic                    

                total_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # validation
            pred = torch.max(net(inputs).detach(), dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()

            # logging two types loss and total loss
            running_loss += loss.item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('[MILossFC/Train] Iter: {}, Acc: {:.3f}, Loss: {:.3f}'.format(
                    batch_idx, # Iter
                    100.*correct / total, # Acc
                    running_loss / (batch_idx+1) # CrossEntropy
                    )
                )

                if args.info_plane:
                    compute_hsic(info_x, info_y, current_iteration)
                    current_iteration += 1


        # Scheduling learning rate by stepLR
        scheduler.step()

        # Adversarial validation
        test()
        adv_test()
        
        # Save checkpoint file
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'running_loss' : running_loss / (batch_idx+1),
            }, os.path.join(args.save_dir,checkpoint_name))
        
        # argument print
        argument_print(args, checkpoint_name)

    # write info
    if args.info_plane or args.adv_plane:
        with open(args.save_IP, 'wb') as f:
            np.save(f, info)
        with open(args.save_fc, 'wb') as f:
            np.save(f, info_fc)
    # with open(args.save_LL, 'wb') as f:
    #     np.save(f, info_tt)
    


def test():

    correct = 0
    total = 0
    net.eval()
    print('\n\n[Natural/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()

        # Evaluation
        outputs = net(inputs).detach()

        # Test
        predicted = torch.max(outputs, dim=1)[1]
        total += targets.numel()
        correct += (predicted == targets).sum().item() 
        
    print('[Natural/Test] Acc: {:.3f}'.format(100.*correct / total))


def adv_test():
    correct = 0.0
    total = 0
    # validation loop

    net.eval()
    print('\n\n[Adv/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()

        adv_input = attack(inputs, targets)
        pred = net(adv_input)

        _, predicted = torch.max(pred, dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item() 


    print('[Adv/Test] Acc: {:.3f}'.format(100.*correct / total))


def pre_infodata():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        for i in range(args.batch_size):
            if i+batch_idx*args.batch_size == num_test:
                return
            info_x[i+batch_idx*args.batch_size] = inputs[i]
            info_y[i+batch_idx*args.batch_size][targets[i]] = 1


def mi_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):
    var = 1e-1
    mi_hy_val = mi( hidden, h_target, var)
    mi_hx_val = mi( hidden, h_data, var)

    return mi_hx_val, mi_hy_val


def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):
    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)

    return hsic_hx_val, hsic_hy_val


def compute_hsic(X, Y, current_iteration):
    net.eval()
    
    if args.adv_plane:
        targets = torch.topk(Y, 1)[1].squeeze(1) # one hot to classes
        X = attack(X, targets)

        net.record = True
        outputs, intermediates = net(X)
    else:
        net.record = True
        outputs, intermediates = net(X)

    x = X.view(X.shape[0], -1)
    y = Y

    for i in range(len(intermediates)):
        t_i = intermediates[i].view(intermediates[i].shape[0], -1)
        mi_xt = hsic_normalized_cca(x, t_i, sigma=5, k_type_y='gaussian')
        mi_yt = hsic_normalized_cca(y.float(), t_i, sigma=5)

        info[current_iteration][0][i] = mi_xt
        info[current_iteration][1][i] = mi_yt

        # if i < (len(intermediates) - 1):
        #     t_i_next = intermediates[i+1].view(-1, np.prod(intermediates[i+1].size()[1:]))
        #     tl_tll = hsic_normalized_cca(t_i, t_i_next, sigma=5, k_type_y='linear')
        #     mi_tt.append(tl_tll)

    # print('intermediates[id_last_cov]: ', intermediates[id_last_cov].shape)
    compute_FC(x, y, intermediates[id_last_cov], current_iteration)


def compute_FC(X, Y, T, current_iteration):
    '''
    calculate mi on different features.
    T is the output of the last convolution layer.
    '''
    x = X.view(X.shape[0], -1)
    y = Y
    num_fc = T.shape[1] # number of feature channel

    for i in range(num_fc-1):
        fc_i = T[:,i:i+1].view(T.shape[0], -1)

        mi_xt = hsic_normalized_cca(x, fc_i, sigma=5)
        mi_yt = hsic_normalized_cca(y.float(), fc_i, sigma=5)

        info_fc[current_iteration][0][i] = mi_xt
        info_fc[current_iteration][1][i] = mi_yt

        # print('xt, yt: ', mi_xt, mi_yt)
    

def compute_MI():
    var = 1e-1
    num_tdata = 200 # take 200 pieces of test data to compute mutual infomation
    net.eval()
    net.record = True
    outputs, intermediates = net(info_x[0:num_tdata].cuda())
    mi_xyt= []

    x = info_x.view(-1, np.prod(info_x.size()[1:]))[0:num_tdata]
    y = info_y[0:num_tdata]

    # print('-----------')

    for i in range(len(intermediates)):
        t_i = intermediates[i].view(-1, np.prod(intermediates[i].size()[1:]))
        mi_xt = mi(x, t_i, var)
        mi_yt = mi(y.float(), t_i, var)
        mi_xyt.append((mi_xt, mi_yt))
        print('mi_xt: {}, mi_yt: {}'.format(mi_xt, mi_yt))
    return mi_xyt


if __name__ == "__main__":
    train()
