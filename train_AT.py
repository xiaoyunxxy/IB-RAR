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
from hsic import *
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
parser.add_argument('--save_IP', default='./at.npy', type=str, help='info plane save directory')
parser.add_argument('--save_LL', default='./at_ll.npy', type=str, help='info between layers save directory')
parser.add_argument('--save_fc', default='./mi_fc.npy', type=str, help='info for feature channels save directory')
parser.add_argument('--info_plane', default=False, type=str2bool, help='compute IP')
parser.add_argument('--adv_plane', default=False, type=str2bool, help='compute IP of adv examples')
parser.add_argument('--fc', default=False, type=str2bool, help='feature channels')
parser.add_argument('--advinput', default=False, type=str2bool, help='use adv input to compute MI for loss')
parser.add_argument('--sf', default=False, type=str2bool, help='use MI between every 2 classes')
parser.add_argument('--selec_layer', default='all', type=str, help='input 1,2,3 etc. Selecting layers for mutual information')

# parser.add_argument('--prior_datetime', default='05070318', type=str, help='checkpoint datetime')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

# loading dataset, network, and attack
trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
net.train()

# checkpoint_name = 'Plain'+'_'+args.network+'_'+args.dataset+'_'+args.prior_datetime+'.pth'
# print('[AT] ' + checkpoint_name +' has been Successfully Loaded')
# state_dict = torch.load(os.path.join(args.save_dir, checkpoint_name))['model_state_dict']
#
# # remove module name
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
checkpoint_name = 'AT_'+args.network+'_'+args.dataset+'_'+date_time+'.pth'

# argument print
argument_print(args, checkpoint_name)

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
    # pre_infodata()

    # Modeling Adversarial Loss
    for epoch in range(args.epoch):

        # train environment
        

        print('\n\n[AT/Epoch] : {}'.format(epoch+1))

        total_cross_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            net.train()
            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.cuda(), targets.cuda()

            # generating adversarial examples
            adv_x = attack(inputs, targets) if args.eps != 0 else inputs

            # learning network parameters
            optimizer.zero_grad()
            net.record = True

            if args.fc and batch_idx % 10 == 0 and batch_idx != 0:
                net.targets = torch.nn.functional.one_hot(targets)

            adv_logit, intermediates = net(adv_x)
            if args.selec_layer != 'all':
                intermediates = layer_selection(intermediates, args.selec_layer)
                
            cross_entropy_loss = F.cross_entropy(adv_logit, targets)

            if args.mi_loss:
                total_loss = cross_entropy_loss
                h_target = torch.nn.functional.one_hot(targets)
                if args.advinput:
                    h_data = adv_x.view(adv_x.shape[0], -1)
                else:
                    h_data = inputs.view(inputs.shape[0], -1)
                    
                for t_index in range(len(intermediates)):
                    intermediates[t_index] = intermediates[t_index].view(
                        intermediates[t_index].shape[0], -1)

                    hx_l, hy_l = hsic_objective(
                        intermediates[t_index],
                        h_target=h_target.float(),
                        h_data=h_data,
                        sigma=5,
                        k_type_y='linear'
                        )

                    temp_hsic = args.lx * hx_l - args.ly * hy_l
                    total_loss += temp_hsic

                if args.sf and batch_idx % 20 == 0 and batch_idx != 0:
                    if args.network == 'resnet18':
                        layer_n = -1
                    elif args.network == 'wide':
                        layer_n = -1
                    else:
                        layer_n = 4
                    MI_class = class_data_MI(layer_n=layer_n)

                    for mic in MI_class:
                        total_loss += mic * 0.001

                total_loss.backward()
            else:
                if args.sf and batch_idx % 20 == 0 and batch_idx != 0:
                    if args.network == 'resnet18':
                        layer_n = -1
                    elif args.network == 'wide':
                        layer_n = -1
                    else:
                        layer_n = 4
                    MI_class = class_data_MI(layer_n=layer_n)

                    for mic in MI_class:
                        cross_entropy_loss += mic * 0.001

                cross_entropy_loss.backward()

            optimizer.step()

            # validation
            pred = torch.max(net(adv_x).detach(), dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()

            # logging two types loss and total loss
            if args.mi_loss:
                total_cross_loss += total_loss.item()
            else:
                total_cross_loss += cross_entropy_loss.item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('[AT/Train] Iter: {}, Acc: {:.3f}, Loss: {:.3f}'.format(
                    batch_idx, # Iter
                    100.*correct / total, # Acc
                    total_cross_loss / (batch_idx+1) # CrossEntropy
                    )
                )
                if args.info_plane:
                    compute_hsic(info_x, info_y, current_iteration)
                    current_iteration += 1

        # Scheduling learning rate by stepLR
        scheduler.step()

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


def adversarial_test():

    correct = 0
    total = 0
    print('\n\n[AT/Test] Under Testing ... Wait PLZ')
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
    print('[AT/Test] Acc: {:.3f}'.format(100.*correct / total))


def class_data_MI(layer_n=4):
    '''
    Take 100 images from each class.
    compute the MI between every 2 classes. 
    '''

    res_mi = []

    take_num = 20 # take 100 image from every class

    c_data = torch.zeros([10, take_num, 3, 32, 32]).cuda()
    count_inte = torch.zeros([10]).int().cuda()
    c_targets = torch.ones([10, take_num]).cuda()

    for i in range(10):
        c_targets[i] *= i

    c_targets = c_targets.long()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        if count_inte.sum() == take_num*10:
            break

        for i in range(args.batch_size):
            if count_inte[targets[i]] >= take_num:
                break

            c_data[targets[i]][count_inte[targets[i]].item()] = inputs[i]
            count_inte[targets[i]] += 1

    for i in range(10):
        for j in range(i, 10):
            if i==j:
                continue

            adv_input_i = attack(c_data[i], c_targets[i])
            adv_input_j = attack(c_data[j], c_targets[j])

            net.record = True
            outputs_i, intermediates_i = net(adv_input_i)
            net.record = True
            outputs_j, intermediates_j = net(adv_input_j)

            intermediates_i[layer_n] = intermediates_i[layer_n].view(intermediates_i[layer_n].shape[0], -1)
            intermediates_j[layer_n] = intermediates_j[layer_n].view(intermediates_j[layer_n].shape[0], -1)

            tmp_mi = hsic_normalized_cca(intermediates_i[layer_n], intermediates_j[layer_n], sigma=5)
            res_mi.append(tmp_mi)

    return res_mi


def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):
    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)

    return hsic_hx_val, hsic_hy_val


def pre_infodata():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        for i in range(args.batch_size):
            if i+batch_idx*args.batch_size == num_test:
                return
            info_x[i+batch_idx*args.batch_size] = inputs[i]
            info_y[i+batch_idx*args.batch_size][targets[i]] = 1


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
    

if __name__ == "__main__":
    train()