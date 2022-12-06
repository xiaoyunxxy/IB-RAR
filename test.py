
# numpy package
import numpy as np

# torch package
import torch
import torchvision
from torchvision import transforms
from torchvision import utils as vutils
from torch.nn.functional import cross_entropy, softmax
import torch.nn.functional as F
from hsic import hsic_normalized_cca
from adp_attack import pgd_mi

# basic package
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import shap

# custom package
from loader.loader import dataset_loader, network_loader, attack_loader

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


parser = argparse.ArgumentParser(description='Adversarial Test')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--data_root', default='../data', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--model_dir', default='./experiment/Plain_vgg16_cifar10_07131357.pth', type=str, help='save directory')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--network', default='vgg16', type=str, help='network name')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--fc', default=False, type=str2bool, help='feature channels')
parser.add_argument('--cwsteps', default=200, type=int, help='adv. steps')
parser.add_argument('--attack_mi', default=False, type=str2bool, help='using MI in PGD')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id


# loader trained network
args.mean=0.5
args.std=0.25
trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
net.load_state_dict(torch.load(args.model_dir)['model_state_dict'])

# attack
attack = attack_loader(args, net)
if args.attack_mi:
    attack = pgd_mi.PGD(model=net, eps=args.eps,
                                alpha=args.eps/args.steps*2.3, steps=args.steps, random_start=True)
    
classes = ('plane', 'car', 'bird', 'cat',
       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def test():
    correct = 0
    total = 0
    conf_sum = 0.0
    soft_conf_sum = 0.0

    net.eval()
    print('\n\n[Plain/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()

        # Evaluation
        outputs = net(inputs).detach()

        # Test
        conf_values, predicted = torch.max(outputs, dim=1)
        conf_sum += conf_values.sum().item()
        total += targets.numel()
        correct += (predicted == targets).sum().item() 

        soft_out = softmax(outputs, dim=1)
        conf_values, predicted = torch.max(soft_out, dim=1)
        soft_conf_sum += conf_values.sum().item()
        
    print('[Plain/Test] Acc: {:.3f}'.format(100.*correct / total))
    print('Confidence value: ', conf_sum / total)
    print('Soft max Confidence value: ', soft_conf_sum / total)
    return 100.*correct / total


def adv_test():
    correct = 0.0
    total = 0
    conf_sum = 0.0
    soft_conf_sum = 0.0
    # validation loop

    net.eval()
    print('\n\n[Plain/Adv_Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()

        adv_input = attack(inputs, targets)
        pred = net(adv_input)

        conf_values, predicted = torch.max(pred, dim=1)
        conf_sum += conf_values.sum().item()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        soft_out = softmax(pred, dim=1)
        conf_values, predicted = torch.max(soft_out, dim=1)
        soft_conf_sum += conf_values.sum().item()


    print('[Adv/Test] Acc: {:.3f}'.format(100.*correct / total))
    print('Confidence value: ', conf_sum / total)
    print('Soft max Confidence value: ', soft_conf_sum / total)
    return 100.*correct / total


def adv_fc():
    '''
    check the difference of feature channel between 
    adversarial example and natural data
    '''
    net.eval()
    print('\n\n Difference of feature channel...')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx <= 50:
            continue
        inputs, targets = inputs.cuda(), targets.cuda()

        adv_input = attack(inputs, targets)
        net.record = True

        if args.fc:
            net.targets = torch.nn.functional.one_hot(targets)
        adv_outputs, adv_intermediates = net(adv_input)

        net.record = True
        outputs, intermediates = net(inputs)
        print('target: ', targets)
        # vgg, use outputs of 4th layer
        t = 352
        for i in range(args.batch_size):
            dif = intermediates[4][i] - adv_intermediates[4][i]
            sorted_dif, indices = torch.sort(torch.abs(dif.flatten()))
            if t in indices[-20:,]:
                print(targets[i])
        return 


def adv_class():
    '''
    check the ground truth target and the adv miss-classified target.
    '''
    net.eval()
    pred_count = torch.zeros([10, 10])
    cofi_count = torch.zeros([10, 10])

    print('Classification Tendency...')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        adv_input = attack(inputs, targets)
        adv_outputs = net(adv_input)
        soft_out = softmax(adv_outputs, dim=1)
        conf_values, predicted = torch.max(soft_out, dim=1)

        for i in range(args.batch_size):
            # print('pred and truth targets: ', targets[i].item(), '  ', predicted[i].item())
            pred_count[targets[i].item()][predicted[i].item()] += 1
            cofi_count[targets[i].item()][predicted[i].item()] += conf_values[i].item()

    for i in range(pred_count.shape[0]):
        tmp, indices = torch.topk(pred_count[i], 4)
        print(classes[i], '&\t', end='')
        for j in range(indices.shape[0]):
            # with probability
            # print(classes[indices[j]], '-', tmp[j].item(), '-', 
            #     format(cofi_count[i][indices[j]].item()/tmp[j].item(), '.3f'), '\t', end='')
            # without probability
            print(classes[indices[j]], '-', tmp[j].int().item(), ' & ', end='')
        print('\n')


# ------- explanation -------
def do_explanation():
    topk = 4
    num_test = 5
    start_test = 20

    target_model = 'AT_vgg16_cifar10_10061732.png'


    for batch_idx, (inputs, targets) in enumerate(testloader):


        adv_input = attack(inputs, targets)
        adv_images = adv_input.cpu()

        images = inputs.cpu()
        netc = net.cpu()
        background = images[:100]

        classes = ['plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        to_explain = images[start_test:(start_test+num_test)]

        # load the ImageNet class names
        class_names = classes

        print('explaining...')
        outputs = net(inputs[start_test:(start_test+num_test)])
        soft_out = softmax(outputs, dim=1)
        conf_values, indices = torch.topk(soft_out, topk)
        print('softmax probability \n ')
        for i in range(indices.shape[0]):
            print('----------')
            for j in range(indices[i].shape[0]):
                print(classes[indices[i][j]], ': ', format(conf_values[i][j].item(), '.3f'))


        # natural example
        e = shap.GradientExplainer((netc, netc.layer2 [0][0]), background)
        
        shap_values,indexes = e.shap_values(to_explain, ranked_outputs=topk, nsamples=200)

        # # get the names for the classes
        index_names = np.vectorize(lambda x: class_names[int(x)])(indexes)

        # # plot the explanations
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

        shap.image_plot(shap_values, to_explain.permute(0,2,3,1).cpu().numpy(), index_names)
        plt.savefig(os.path.join('./exfig', target_model))


        # adv -------
        adv_to_explain = adv_images[start_test:(start_test+num_test)]
        e = shap.GradientExplainer((netc, netc.layer2[0][0]), background)
        print('\n\nadv explaining...')
        shap_values,indexes = e.shap_values(adv_to_explain, ranked_outputs=topk, nsamples=200)

        # # get the names for the classes
        index_names = np.vectorize(lambda x: class_names[int(x)])(indexes)

        # # plot the explanations
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

        shap.image_plot(shap_values, adv_to_explain.permute(0,2,3,1).cpu().numpy(), index_names)
        plt.savefig(os.path.join('./exfig', 'adv_'+target_model))


        print('softmax probability \n ')
        netc = net.cuda()
        outputs = netc(adv_input[start_test:(start_test+num_test)])
        soft_out = softmax(outputs, dim=1)
        conf_values, indices = torch.topk(soft_out, topk)
        
        for i in range(indices.shape[0]):
            print('----------')
            for j in range(indices[i].shape[0]):
                print(classes[indices[i][j]], ': ', format(conf_values[i][j].item(), '.3f'))

        break
        
# ------- end of explanation -------


def shared_feature_check():
    '''
    check the MI of 2 different classes
    '''
    classes_pair = [(0, 2), (1, 9), (2, 4), (3, 5), (4, 2), (5, 3), (6, 3), (7, 5), (8, 0), (9, 1)]

    take_num = 10

    layer_n = 4

    tem_inte = torch.zeros([10, take_num, 512]).cuda()
    count_inte = torch.zeros([10]).int().cuda()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_input = attack(inputs, targets)

        net.record = True
        outputs, intermediates = net(adv_input)
        intermediates[layer_n] = intermediates[layer_n].view(intermediates[layer_n].shape[0], -1)
            
        if count_inte.sum() == take_num*10:
            break

        for i in range(args.batch_size):
            if count_inte[targets[i]] >= take_num:
                break

            tem_inte[targets[i]][count_inte[targets[i]].item()] = intermediates[layer_n][i]
            count_inte[targets[i]] += 1

    for i in range(10):
        print('-- -- --')
        one_class = []
        for j in range(10):
            tmp_mi = hsic_normalized_cca(tem_inte[i], tem_inte[j], sigma=5)
            
            if  i != j:
                one_class.append(tmp_mi.item())

            print(classes[i], '\t\t', classes[j], '\t\t', format(tmp_mi.item(), '.3f'))

        print('variance: ', np.var(one_class))


attack_list = ['pgd', 'cw', 'fgsm', 'fab', 'nifgsm']
def multiple_adv_test(al=attack_list):
    acc = []
    acc_natural = test()
    acc.append(acc_natural)
    global attack
    for i in al:
        args.attack = i
        loginfo = '\n\n---- test ' + i + ' attack ----'
        print(loginfo)
        attack = attack_loader(args, net)
        adv_acc = adv_test()
        acc.append(adv_acc)
    print(acc)


def pgd_mutiple_steps():
    step_list = [1, 10, 20, 30, 40, 50]
    # fgsm at first
    acc = []
    global attack
    args.attack = 'fgsm'
    attack = attack_loader(args, net)
    adv_acc = adv_test()
    acc.append(adv_acc)
    args.attack = 'pgd'

    for i in step_list:
        args.steps = i
        attack = attack_loader(args, net)
        adv_acc = adv_test()
        acc.append(adv_acc)

    print(acc)


def cw_mutiple_steps():
    step_list = [10, 20, 30, 40, 50]
    acc = []
    global attack
    args.attack = 'cw'
    
    for i in step_list:
        args.cwsteps = i
        attack = attack_loader(args, net)
        adv_acc = adv_test()
        acc.append(adv_acc)
    print(acc)

def nifgsm_mutiple_steps():
    step_list = [1, 3, 5, 7, 9, 10, 20]
    acc = []
    global attack
    args.attack = 'nifgsm'
    
    for i in step_list:
        args.steps = i
        attack = attack_loader(args, net)
        adv_acc = adv_test()
        acc.append(adv_acc)
    print(acc)


if __name__ == '__main__':
    # test()
    adv_test()
    # multiple_adv_test()
    # nifgsm_mutiple_steps()
    # do_explanation()
    # adv_class()
    # shared_feature_check()
    
