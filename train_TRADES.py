#!/usr/bin/env python

# torch package
import torch
import torch.nn.functional as F

# basic package
import os
import argparse
from tqdm import tqdm
from datetime import datetime

# custom package
from loader.argument_print import argument_print
from loader.loader import dataset_loader, network_loader, attack_loader
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
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--network', default='vgg16', type=str, help='network name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='../data', type=str, help='path to dataset')
parser.add_argument('--epoch', default=60, type=int, help='epoch number')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--save_dir', default='/data/xuxx/experiment_MI/', type=str, help='save directory')
parser.add_argument('--mi_loss', default=False, type=str2bool, help='use mi loss or not')
parser.add_argument('--lx', default=0.008, type=float, help='regular for I(X,T)')
parser.add_argument('--ly', default=0.08, type=float, help='regular for I(Y,T)')
parser.add_argument('--fc', default=False, type=str2bool, help='feature channels')
parser.add_argument('--advinput', default=False, type=str2bool, help='use adv input to compute MI for loss')
parser.add_argument('--selec_layer', default='all', type=str, help='input 1,2,3 etc. Selecting layers for mutual information')

# parser.add_argument('--prior_datetime', default='00000000', type=str, help='checkpoint datetime')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

# loading dataset, network, and attack
trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
net.train()

# checkpoint_name = 'Plain'+'_'+args.network+'_'+args.dataset+'_'+args.prior_datetime+'.pth'
# print('[TRADE] ' + checkpoint_name +' has been Successfully Loaded')
# state_dict = torch.load(os.path.join(args.save_dir, checkpoint_name))['model_state_dict']
# #
# # # remove module name
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# state_dict = new_state_dict
# net.load_state_dict(state_dict)

if len(args.gpu_id.split(','))!=1:
    net = torch.nn.DataParallel(net)
attack = attack_loader(args, net)

# Adam Optimizer with KL divergence, and Scheduling Learning rate
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# Setting checkpoint date time
date_time = datetime.today().strftime("%m%d%H%M")

# checkpoint_name
checkpoint_name = 'TRADE_'+args.network+'_'+args.dataset+'_'+date_time+'.pth'

# argument print
argument_print(args, checkpoint_name)

# Loss function
criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

def train():

    # Modeling Adversarial Loss
    for epoch in range(args.epoch):

        # train environment
        net.train()

        print('\n\n[TRADE/Epoch] : {}'.format(epoch+1))

        total_cross_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.cuda(), targets.cuda()

            # generating adversarial examples
            adv_x = attack(inputs, targets) if args.eps != 0 else inputs

            # learning network parameters
            optimizer.zero_grad()

            net.record = True
            if args.fc and batch_idx % 10 == 0 and batch_idx != 0:
                net.targets = torch.nn.functional.one_hot(targets)
            logits, intermediates = net(inputs)
            if args.selec_layer != 'all':
                intermediates = layer_selection(intermediates, args.selec_layer)

            net.record = True
            if args.fc and batch_idx % 10 == 0 and batch_idx != 0:
                net.targets = torch.nn.functional.one_hot(targets)
            logits_adv, adv_intermediates = net(adv_x)
            if args.selec_layer != 'all':
                adv_intermediates = layer_selection(adv_intermediates, args.selec_layer)

            loss_natural = F.cross_entropy(logits_adv, targets)
            loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
            loss = loss_natural + 0.5 * loss_robust

            if args.mi_loss:
                total_loss = loss
                intermediates += adv_intermediates
                h_target = torch.nn.functional.one_hot(targets)
                if args.advinput:
                    h_data = adv_x.view(adv_x.shape[0], -1)
                else:
                    h_data = inputs.view(inputs.shape[0], -1)
                    
                for t_index in range(len(intermediates)):
                    intermediates[t_index] = intermediates[t_index].view(intermediates[t_index].shape[0], -1)

                    hx_l, hy_l = hsic_objective(
                        intermediates[t_index],
                        h_target=h_target.float(),
                        h_data=h_data,
                        sigma=5
                        )

                    temp_hsic = args.lx * hx_l - args.ly * hy_l
                    total_loss += temp_hsic

                total_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # validation
            pred = torch.max(net(adv_x).detach(), dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()

            # logging two types loss and total loss
            if args.mi_loss:
                total_cross_loss += total_loss.item()
            else:
                total_cross_loss += loss.item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('[TRADE/Train] Iter: {}, Acc: {:.3f}, CE: {:.3f}'.format(
                    batch_idx, # Iter
                    100.*correct / total, # Acc
                    total_cross_loss / (batch_idx+1) # CrossEntropy
                    )
                )

        # Scheduling learning rate by stepLR
        scheduler.step()

        # torch.cuda.empty_cache()
        
        # Adversarial validation
        adversarial_test()
        test()
        # Save checkpoint file
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_cross_entropy_loss' : total_cross_loss / (batch_idx+1)
            }, os.path.join(args.save_dir,checkpoint_name))

        # argument print
        argument_print(args, checkpoint_name)


def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):
    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)

    return hsic_hx_val, hsic_hy_val


def adversarial_test():

    correct = 0
    total = 0
    print('\n\n[TRADE/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_x = attack(inputs, targets) if args.eps != 0 else inputs

        # Evaluation
        net.eval()
        outputs = net(adv_x).detach()

        # Test
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print('[TRADE/Test] Acc: {:.3f}'.format(100.*correct / total))

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


if __name__ == "__main__":
    train()
