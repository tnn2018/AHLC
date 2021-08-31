from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import models
import operator

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=0.5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./prune', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='resnet', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=56, type=int,
                    help='depth of the neural network')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')      

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 


torch.backends.cudnn.benchmark = True

torch.manual_seed(args.seed)
if args.cuda:
    print('using cuda')
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

batch_size = args.batch_size

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    
    indices = list(range(50000))
    split = int(np.floor(45000))
    np.random.seed(1)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    print(len(train_indices))
    print(len(val_indices))
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size,  sampler=train_sampler, shuffle=False, **kwargs)
    print(len(train_loader)) 
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size,  sampler=valid_sampler, shuffle=False, **kwargs)
    print(len(valid_loader))    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),    
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    indices = list(range(50000))
    split = int(np.floor(45000))
    np.random.seed(1)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    print(len(train_indices))
    print(len(val_indices))
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, sampler=train_sampler, shuffle=False, **kwargs)
    print(len(train_loader)) 
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, sampler=valid_sampler, shuffle=False, **kwargs)
    print(len(valid_loader)) 
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# ******************************
weight_p, bias_p = [],[]
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

lambda_1 = torch.tensor(0.5) 
weight_decay = args.weight_decay

test_acc_all = []  
val_acc_all  = []         
# *****************************
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    
def train(epoch, lambda_1):
    print(lambda_1, weight_decay)
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item() 
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()   
        # **************
        #for group in self.param_groups:
        #    for p in group['params']:
        #        if p.grad is None:
        #            continue
        #        d_p = p.grad.data
        #        if weight_decay != 0:
        #            d_p.add_(weight_decay, p.data)
        #        p.data.add_(-group['lr'], d_p)        
       # for group in optimizer.param_groups:  
        num = 0         
        for p in weight_p:   
                
           # for p in group['params']:
               # print(p)
               # print(p.grad)                  
                
                    #print('pass')
                if p.grad is not None: 
                   
                   k1 = torch.tensor(0.)
                  # print(k1)
                   k2 = torch.zeros(p.shape)
                  # print(k2)
                   f1 = operator.gt(p.cpu(),k1)
                   f2 = operator.lt(p.cpu(),k2)
                   f1 = f1.float()
                   f2 = f2.float()
                   f =  f1 - f2
                   f = f/1000000

                  # print(f)
                  # print('ss*********************************sss')
                   p.grad += (lambda_1*f).cuda() + weight_decay*p/10000
                   if operator.gt(epoch,40) == 1 : 
                      f3 = operator.lt(p.cpu().abs(), torch.tensor(1e-3)) 
                      f4 = torch.zeros(p.data.shape).cuda()
                      #print(f3)
                      #print(p.data[f3]) 
                      p.data = torch.where( p.data.abs() < 1e-4, f4, p.data)
                      #print(p.data)
                      #print(torch.sum(operator.eq(p.cpu(),f4.cpu())))
                      #print(p.data)
                      p.grad = torch.where(p.data == 0., f4, p.grad) 
                      #print(p.grad)
                      #weight_p[num] = p
                      #p.grad[f3] = 0 
                      #print(p.data[f3]) 
                      #print(p.data)
                      #print('pass')
                      num +=  1 
        #print('********num:') 
        #print(num)            
        # ***************
        #loss.backward()
        optimizer.step()
        # **********************   
        zero_num = torch.tensor(0.0000)
        for p in weight_p:
            #print(p.data)
            k = torch.zeros(p.shape).cuda()
            zero_num += torch.sum(operator.eq(p.data,k))
       # zero_rate = zero_num / total_num   
       # ***********************
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f} zero_num: {}'.format(
                epoch, batch_idx * len(data), len(train_loader)*len(data),
                100. * batch_idx / len(train_loader), loss.item(),zero_num)) 
    return weight_p                  
        # *********************    
def val(epoch, weight_q, lambda_1, weight_decay):
    print(lambda_1, weight_decay)
    optimizer = optim.SGD([
                {'params': weight_p, 'weight_decay':0},
                {'params': bias_p, 'weight_decay':0}
               ], lr=args.lr, momentum=args.momentum ) 
    model.eval()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item() 
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        
        # ***********************
        theta = torch.tensor(1e-5)
        #lambda_1_2 = Variable(torch.tensor(0., 0.))
        lambda_1_2 = torch.tensor([0., 0.])
        count = 0
        for p in weight_p: 
            q = weight_q[count]
            if q.grad is not None: 
               k = q.grad.view(-1,1) 
               k1 = torch.ones(k.size(0),1) 
               k2 = torch.cat((k1, q.cpu().view(-1,1)), 1)
               if p.grad is not None:
                  k3 = p.grad.view(1,-1) 
                  k4 = k3.cpu().mm(k2)
                  k4 = k4.squeeze(0)
                  lambda_1_2 += k4
                  
                  #print('success')
            count += 1
        lambda_1 += (theta)*lambda_1_2[0]
        weight_decay += (theta)*lambda_1_2[1]
        lambda_1 = lambda_1.data
        weight_decay = weight_decay.data 
        #print(lambda_1, weight_decay)
        # ***********************
        #optimizer.step()
        if batch_idx % 10 == 0: 
            print('valid Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f} lambda_1_2.grad: {} lambda_1: {} weight_decay: {}'.format(
                epoch_v, batch_idx * len(data), len(valid_loader)*len(data),
                100. * batch_idx / len(valid_loader), loss.item(), lambda_1_2, lambda_1, weight_decay))
    #print(lambda_1, weight_decay)         
    return  lambda_1, weight_decay 
          


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
           if args.cuda:
              data, target = data.cuda(), target.cuda()
           data, target = Variable(data), Variable(target)
           output = model(data)
    
           test_loss += F.cross_entropy(output, target, reduction='sum').item()  # new 
           pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
           correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        
        correct = correct.numpy() 
        correct = float(correct / 100) 
        return correct 

def valtest(): 
    model.eval()
    valid_loss = 0
    correct = 0   
    with torch.no_grad():
        for data, target in valid_loader:
           if args.cuda:  
              data, target = data.cuda(), target.cuda()
           data, target = Variable(data), Variable(target)
           output = model(data)
    
           valid_loss += F.cross_entropy(output, target, reduction='sum').item()  # new 
           pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
           correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        valid_loss /= 5000

        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                valid_loss, correct,5000,
                 100. * correct/float(5000.0) ))
       
        correct = correct.numpy() 
        correct = float(correct / 50) 
        return correct 

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


for epoch_all in range(4):
    print(epoch_all)
    #optimizer = optim.SGD([
    #            {'params': weight_p, 'weight_decay':weight_decay},
    #            {'params': bias_p, 'weight_decay':0}
    #           ], lr=args.lr, momentum=args.momentum ) 
    optimizer = optim.SGD([
                {'params': weight_p, 'weight_decay':0},
                {'params': bias_p, 'weight_decay':0}
               ], lr=args.lr, momentum=args.momentum )  
    best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        weight_p = train(epoch, lambda_1=lambda_1)
        prec1 = test()
        test_acc_all.append(prec1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'cfg': model.cfg
        }, is_best, filepath=args.save)
    for epoch_v in range(2): 
        lambda_1, weight_decay = val(epoch=epoch_v, weight_q=weight_p, lambda_1=lambda_1, weight_decay=weight_decay)
        prec2 = valtest()
        val_acc_all.append(prec2)
test_acc_all = np.asarray(test_acc_all)
val_acc_all = np.asarray(val_acc_all)    
scipy.io.savemat('./p5res56/p5test_acc_all.mat', mdict={'p5test_acc_all': test_acc_all})
scipy.io.savemat('./p5res56/p5val_acc_all.mat', mdict={'p5val_acc_all': val_acc_all})        