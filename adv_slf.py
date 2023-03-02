from __future__ import print_function

import argparse
import math
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from autoattack import AutoAttack
from torch import nn, optim
from torchvision import datasets, transforms

# from models.resnet_cifar import ResNet18
from solo.models.resnet_cifar import ResNet18, ResNet50
from solo.models.wide_resnet import wide_resnet28w10
from solo.models.model_with_linear import ModelwithLinear, LinearClassifier
from solo.models.resnet_add_normalize import resnet18_NormalizeInput

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_loader(opt):
    # construct data loader
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                     transform=train_transform,
                                     download=True)
    val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                   train=False,
                                   transform=val_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(
            train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10'], help='dataset')
    # other setting
    parser.add_argument('--ckpt', type=str, default='trained_models/res18_simclr-cifar10-offline-x4h7cp45-ep=99.ckpt',
                        help='path to pre-trained model')

    parser.add_argument('--name', type=str, default='deacl_slf',
                        help='name of the exp')

    parser.add_argument('--eval_freq', type=int, default=5,
                        help='print frequency')
    
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './data/'
    opt.n_cls = 10

    return opt


# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, classifier, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.classifier = classifier
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, inputs, targets, train=True):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        if train:
            num_step = 10
        else:
            num_step = 20
        for i in range(num_step):
            x.requires_grad_()
            with torch.enable_grad():
                features = self.model(x)
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon),
                          inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        features = self.model(x)
        return self.classifier(features), x


def set_model(opt):
    if "res50" in opt.ckpt:
        model = ResNet50()
        classifier = LinearClassifier(
            name=opt.name, feat_dim=2048, num_classes=opt.n_cls)
    elif "wideres28_10" in opt.ckpt:
        model = wide_resnet28w10()
        classifier = LinearClassifier(name=opt.name, feat_dim=model.inplanes, num_classes=opt.n_cls)
    else:

        model = resnet18_NormalizeInput()
        model.fc = nn.Identity()
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        model.maxpool = nn.Identity()
        classifier = LinearClassifier(
            name=opt.name, feat_dim=512, num_classes=opt.n_cls)

    criterion = torch.nn.CrossEntropyLoss()

    print('loading from {}'.format(opt.ckpt))
    state_dict = torch.load(opt.ckpt, map_location='cpu')

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    state_dict_load = {}
    for k,v in state_dict.items():
        if k.startswith('backbone.'):
            state_dict_load[k.replace('backbone.', '')] = v.clone()

    model = model.cuda()
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    config = {
        'epsilon': 8.0 / 255.,
        'num_steps': 20,
        'step_size': 2.0 / 255,
        'random_start': True,
        'loss_func': 'xent',
    }
    net = AttackPGD(model, classifier, config)
    net = net.cuda()
    cudnn.benchmark = True

    model.load_state_dict(state_dict_load, strict=True)

    return model, classifier, net, criterion


def train(train_loader, model, classifier, net, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            features = model(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, idx + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, net, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    top1_clean = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output, _ = net(images, labels, train=False)
            # output_feats = model(images)
            # output = classifier(output_feats)

            # import ipdb; ipdb.set_trace()
            loss = criterion(output, labels)

            features_clean = model(images)
            output_clean = classifier(features_clean)
            acc1_clean, acc5_clean = accuracy(
                output_clean, labels, topk=(1, 5))

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top1_clean.update(acc1_clean[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 Clean {top1_clean.val:.4f} ({top1_clean.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          idx, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1, top1_clean=top1_clean))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Acc@1 Clean {top1_clean.avg:.3f}'.format(top1_clean=top1_clean))
    return losses.avg, top1.avg, top1_clean.avg


def adjust_lr(lr, optimizer, epoch):
    if epoch >= 15:
        lr /= 10
    if epoch >= 20:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(slf_config=None, ):
    best_acc = 0
    best_acc_clean = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, net, criterion = set_model(opt)

    # build optimizer
    params = list(classifier.parameters())
    optimizer = optim.SGD(params,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # training routine
    val_acc = 0
    val_acc_clean = 0
    for epoch in range(1, opt.epochs + 1):
        adjust_lr(opt.learning_rate, optimizer, epoch-1)
        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, net, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        
        if epoch % opt.eval_freq == 0:
            # eval for one epoch
            loss, val_acc, val_acc_clean = validate(
                val_loader, model, classifier, net, criterion, opt)
            
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_clean = val_acc_clean
            log_ra = val_acc
            log_ta = val_acc_clean
            state = {
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'rng_state': torch.get_rng_state()
            }

    print('best accuracy: {:.2f}'.format(best_acc))
    print('best accuracy clean: {:.2f}'.format(best_acc_clean))

    log_path = os.path.join(os.path.dirname(opt.ckpt), f"{os.path.basename(opt.ckpt)}_eval_aa_eval_best.log")
    
    model.fc = classifier
    # model = ModelwithLinear(model, model.inplanes)
    # model.classifier.weight = classifier.classifier.weight
    # model.classifier.bias = classifier.classifier.bias
    
    adversary = AutoAttack(model, norm="Linf", eps=8.0 /
                           255., log_path=log_path, version="standard", seed=0)
    l = [x for (x, y) in val_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in val_loader]
    y_test = torch.cat(l, 0)

    with torch.no_grad():
        adversary.run_standard_evaluation(x_test, y_test, bs=256)

    with open(os.path.join(os.path.dirname(opt.ckpt), "final_results.txt"), "a") as f:

        f.write(f"Final robust acc: {val_acc}.\n")
        f.write(f"Final clean acc: {val_acc_clean}.\n")

        f.write(f"Best robust acc: {best_acc}.\n")
        f.write(f"Best clean acc: {best_acc_clean}.\n")




if __name__ == '__main__':
    main()
