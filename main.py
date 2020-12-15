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
    if args.source == None:
        s_train_dataloader, s_test_dataloader = make_dataloader(args)
        t_train_dataloader = s_train_dataloader
        t_test_dataloader = s_test_dataloader
    else:
        s_train_dataloader, s_test_dataloader, t_train_dataloader, t_test_dataloader = make_biased_dataloader(args)
    net = resnet.ResNet34(num_classes=10).to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wdecay)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

    if args.resume == 1:
        checkpoint = torch.load('./checkpoint/'+'RN')
        net.load_state_dict(checkpoint)

    CE_loss = nn.CrossEntropyLoss()

    path = './checkpoint/'+'RN'

    best_acc=0
    for epoch in range(args.epochs):
        train(args, net, s_train_dataloader, t_train_dataloader, optimizer, CE_loss,scheduler, epoch)
        acc = test(args, net, s_test_dataloader, t_test_dataloader, optimizer, CE_loss,scheduler, epoch)
        scheduler.step()
        if best_acc<acc:
            if not os.path.isdir('checkpoint/'):
                os.makedirs('checkpoint/')
            torch.save(net.state_dict(), path)

def train(args,net,s_train_dataloader,t_train_dataloader, optimizer,CE_loss,scheduler,epoch):
    net.train()
    train_loss = 0
    p_bar = tqdm(range(s_train_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (inputs, targets) in enumerate(s_train_dataloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)     
        optimizer.zero_grad()

        outputs,domain_outputs = net(inputs)
        loss = CE_loss(outputs,targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. ce_loss: {ce_loss:.4f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=s_train_dataloader.__len__(),
                    lr=scheduler.get_last_lr()[0],
                    ce_loss = loss,
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/s_train_dataloader.__len__()        # average train_loss

def test(args, net, s_test_dataloader, t_test_dataloader, optimizer, CE_loss, scheduler, epoch):
    net.eval()
    test_loss = 0
    acc = 0.
    p_bar = tqdm(range(s_test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(s_test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs,domain_outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. S_Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=s_test_dataloader.__len__(),
                    lr=scheduler.get_last_lr()[0],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
    p_bar.close()

    acc_t = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(t_test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs, domain_outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. T_Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=s_test_dataloader.__len__(),
                    lr=scheduler.get_last_lr()[0],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc_t+=sum(outputs.argmax(dim=1)==targets)
    p_bar.close()
    
    acc = acc.float()/s_test_dataloader.dataset.__len__()
    acc_t = acc_t.float()/t_test_dataloader.dataset.__len__()
    acc = acc*100
    acc_t = acc_t*100
    print('Source Domain Accuracy :'+ '%0.2f'%acc )
    print('Target Domain Accuracy :'+ '%0.2f'%acc_t)

    return acc

if __name__ == '__main__':
    main()


# TODO : alpha 지정해줘야함