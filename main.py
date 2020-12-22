import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
from utils.arguments import *
from models import resnet
from dataset.distractor import *

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()
    args.device = torch.device('cuda',args.gpu_id)

    #dataset setting
    if args.resume == 1:
        s_train_dataloader, s_test_dataloader, t_train_dataloader, t_test_dataloader = make_biased_dataloader(args)
    elif args.resume == 0:
        s_train_dataloader, s_test_dataloader, t_train_dataloader, t_test_dataloader = make_biased_dataloader(args)

    concat_dataset = ConcatDataset(
        s_train_dataloader.dataset,
        t_train_dataloader.dataset
    )
    concat_dataset_test = ConcatDataset(
        s_test_dataloader.dataset,
        t_test_dataloader.dataset
    )
    train_loader = DataLoader(
        concat_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4
    )
    test_loader = DataLoader(
        concat_dataset_test,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4
    )

    net = resnet.ResNet34(num_classes=10).to(args.device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

    if args.resume == 1:
        checkpoint = torch.load('./checkpoint/Source_Only/'+'RN_'+args.train_mode+'_'+'85')
        net.load_state_dict(checkpoint)

    path = './checkpoint/Source_Only/'+'RN_'+args.train_mode+'_'
    best_acc=0
    for epoch in range(args.epochs):
        train(args, net, train_loader, optimizer, scheduler, epoch)
        acc = test(args, net, test_loader, optimizer, scheduler, epoch)
        scheduler.step()
        if best_acc<acc:
            if not os.path.isdir('checkpoint/Source_Only/'):
                os.makedirs('checkpoint/Source_Only/')
            torch.save(net.state_dict(), path+str(epoch))
            # best_acc = acc/ just save everything

def train(args,net,loader, optimizer, scheduler,epoch):
    net.train()
    train_loss = 0
    p_bar = tqdm(range(loader.__len__()))
    loss_average = 0
    
    for batch_idx, (s_batch, t_batch) in enumerate(loader):
        s_data, s_label = s_batch
        t_data, t_label = t_batch
        s_data = s_data.to(args.device)
        s_label = s_label.to(args.device)
        t_data = t_data.to(args.device)
        t_label = t_label.to(args.device) 
        
        p = float(batch_idx + epoch * loader.__len__()) / args.epochs / loader.__len__()
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        alpha = alpha*0.2
         
        # Source Loss
        s_class_output, s_domain_output = net(s_data,alpha=alpha)
        s_class_loss = F.cross_entropy(s_class_output, s_label)
        s_domain_label = torch.zeros(s_domain_output.shape[0]).long().to(s_domain_output.device)
        s_domain_loss = F.cross_entropy(s_domain_output,s_domain_label)

        # Target Loss
        t_class_output, t_domain_output = net(t_data,alpha=alpha)
        t_class_loss = F.cross_entropy(t_class_output, t_label)
        t_domain_label = torch.ones(t_domain_output.shape[0]).long().to(t_domain_output.device)
        t_domain_loss = F.cross_entropy(t_domain_output,t_domain_label)

        if args.train_mode=='SO':
            loss = s_class_loss
        elif args.train_mode=='CC':
            loss = s_class_loss + t_class_loss
        elif args.train_mode=='DA':
            loss = s_class_loss + 0.2*(s_domain_loss + t_domain_loss)
        elif args.train_mode=='TO':
            loss = t_class_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. class_loss: {class_loss:.4f}. domain_loss: {domain_loss:.4f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=loader.__len__(),
                    lr=scheduler.get_last_lr()[0],
                    class_loss = s_class_loss,
                    domain_loss = s_domain_loss+t_domain_loss,
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/loader.__len__()        # average train_loss

def test(args, net, loader, optimizer, scheduler, epoch):
    net.eval()
    test_loss = 0
    acc_s = 0.
    acc_t = 0.
    p_bar = tqdm(range(loader.__len__()))
    test_class_loss = 0
    test_domain_loss = 0
    with torch.no_grad():
        for batch_idx, (s_batch, t_batch) in enumerate(loader):
            s_data, s_label = s_batch
            t_data, t_label = t_batch
            s_data = s_data.to(args.device)
            s_label = s_label.to(args.device)
            t_data = t_data.to(args.device)
            t_label = t_label.to(args.device)

            p = float(batch_idx + epoch * loader.__len__()) / args.epochs / loader.__len__()
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Source Loss
            s_class_output, s_domain_output = net(s_data,alpha=alpha)
            s_class_loss = F.cross_entropy(s_class_output, s_label)
            s_domain_label = torch.zeros(s_domain_output.shape[0]).long().to(s_domain_output.device)
            s_domain_loss = F.cross_entropy(s_domain_output,s_domain_label)

            # Target Loss
            t_class_output, t_domain_output = net(t_data,alpha=alpha)
            t_class_loss = F.cross_entropy(t_class_output, t_label)
            t_domain_label = torch.ones(t_domain_output.shape[0]).long().to(t_domain_output.device)
            t_domain_loss = F.cross_entropy(t_domain_output,t_domain_label)

            s_class_pred = s_class_output.argmax(dim=1,keepdim=True)
            s_domain_pred = s_domain_output.argmax(dim=1,keepdim=True)
            t_class_pred = t_class_output.argmax(dim=1,keepdim=True)
            t_domain_pred = t_domain_output.argmax(dim=1,keepdim=True)

            class_loss = s_class_loss + t_class_loss
            domain_loss = s_domain_loss + t_domain_loss
            test_class_loss += class_loss.item()
            test_domain_loss += domain_loss.item()
            s_class_correct = s_class_pred.eq(s_label.view_as(s_class_pred)).sum().item()
            s_domain_correct = s_domain_pred.eq(s_domain_label.view_as(s_domain_pred)).sum().item()

            t_class_correct = t_class_pred.eq(t_label.view_as(t_class_pred)).sum().item()
            t_domain_correct = t_domain_pred.eq(t_domain_label.view_as(t_domain_pred)).sum().item()

            p_bar.set_description("Test Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. class_loss: {class_loss:.4f}. domain_loss: {domain_loss:.4f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=loader.__len__(),
                    lr=scheduler.get_last_lr()[0],
                    class_loss = s_class_loss+t_class_loss,
                    domain_loss = s_domain_loss+t_domain_loss,
                    loss = (test_class_loss+test_domain_loss)/(batch_idx+1))
                    )
            p_bar.update()
            acc_s+=s_class_correct
            acc_t+=t_class_correct
    p_bar.close()    
    acc_s = acc_s/loader.dataset.__len__()
    acc_t = acc_t/loader.dataset.__len__()
    acc_s = acc_s*100
    acc_t = acc_t*100
    print('Source Domain Accuracy :'+ '%0.2f'%acc_s )
    print('Target Domain Accuracy :'+ '%0.2f'%acc_t)

    return acc_s

if __name__ == '__main__':
    main()