import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset, valid_dataset
from model_resnet import EAST
from loss import Loss
import os
import time
import numpy as np
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def drawLoss(train_loss, valid_loss, save_name):
    x1 = range(0,len(train_loss))
    x2 = range(0,len(valid_loss))
    # print(x1,":",x2)
    # plt.figure(1)
    plt.plot(x1, train_loss, c='red', label='train loss')
    plt.plot(x2, valid_loss, c='blue', label = 'valid loss')
    plt.xlabel('item number')
    plt.legend(loc='upper left')
    plt.savefig(save_name, format='jpg')


def train(train_img_path, train_gt_path, valid_img_path, valid_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, pretrain=False, model_pth=None, currrent_epoch=0):
    file_num = len(os.listdir(train_img_path))
    valid_file_num = len(os.listdir(valid_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

    validset = valid_dataset(valid_img_path, valid_gt_path)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, \
                                   shuffle=False, num_workers=num_workers, drop_last=False)

    dataLoader = {'train':train_loader, 'valid':valid_loader}

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    if pretrain:
        model.load_state_dict(torch.load(model_pth))
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    res_epoch = epoch_iter - currrent_epoch
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[res_epoch // 4, res_epoch * 2 // 4, res_epoch * 3 // 4], gamma=0.1)

    best_loss = 1000
    best_model_wts = copy.deepcopy(model.state_dict())
    best_num = 0

    train_loss = []
    valid_loss = []
    for epoch in range(epoch_iter):
        if pretrain:
            epoch = currrent_epoch

        for phase in ['train','valid']:
        # for phase in ['valid', 'train']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()
            epoch_loss = 0
            epoch_time = time.time()

            for i, (img, gt_score, gt_geo, ignored_map) in enumerate(dataLoader[phase]):
                start_time = time.time()
                img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred_score, pred_geo = model(img)
                    loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item()


                print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                    epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time, loss.item()))
            epoch_loss_mean = 0
            if phase == 'train':
                epoch_loss_mean = epoch_loss / int(file_num / batch_size)
                train_loss.append(epoch_loss_mean)
            else:
                epoch_loss_mean = epoch_loss / int(valid_file_num / batch_size)
                valid_loss.append(epoch_loss_mean)
            print('phase:{}, epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(phase, epoch_loss_mean ,time.time() - epoch_time))
            print(time.asctime(time.localtime(time.time())))
            print('=' * 50)
            if phase == 'valid' and epoch_loss < best_loss:
                best_num = epoch+1
                best_loss = epoch_loss_mean
                best_model_wts = copy.deepcopy(model.state_dict())
                print('best model num:{}, best loss is {:.8f}'.format(best_num, best_loss))
            if (epoch + 1) % interval == 0 and phase == 'valid':
                savePath = './pths_valid/'+'lossImg'+str(epoch+1)+'.jpg'
                drawLoss(train_loss, valid_loss, savePath)
                print(time.asctime(time.localtime(time.time())))
                state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                lr_state = scheduler.state_dict()
                torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))
                torch.save(lr_state, os.path.join(pths_path, 'scheduler_epoch_{}.pth'.format(epoch + 1)))
                print("save model")
                print('=' * 50)
    # save best model
    torch.save(best_model_wts, os.path.join(pths_path, 'model_epoch_best.pth'))


if __name__ == '__main__':
    # train_img_path = os.path.abspath('../ICDAR_2015/train_img')
    # train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
    train_img_path = '/data/home/zjw/dataset/icdar2015//train_images/'
    train_gt_path = '/data/home/zjw/dataset/icdar2015/train_gts/'
    valid_img_path = '/data/home/zjw/dataset/icdar2015/valid_images/'
    valid_gt_path = '/data/home/zjw/dataset/icdar2015/valid_gts/'
    pths_path = './pths_valid'

    model_pth = '/data/home/zjw/pythonFile/EAST-1/pths_valid/model_epoch_750.pth'
    batch_size = 50
    lr = 1e-3
    num_workers = 4
    epoch_iter = 2000
    save_interval = 1
    train(train_img_path, train_gt_path, valid_img_path,valid_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, True, model_pth, 750)
    # a = [1,2,3,4,5]
    # b = [11,12,13,14,15]
    # drawLoss(a, b, './test.jpg')


