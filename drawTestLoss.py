import time
import torch
import subprocess
import os
from model_resnet import EAST
from detect import detect_dataset
import numpy as np
import shutil
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def drawTestAcc(title, precision, recall, hmean, save_name):

    xP = range(0, len(precision))
    xR = range(0, len(recall))
    xH = range(0, len(hmean))

    # print('XP,XR,XH:'+str(xP)+','+str(xR)+','+str(xH))

    plt.title(title)
    plt.plot(xP, precision, c='red', label='precision')
    plt.plot(xR, recall, c='blue', label='recall')
    plt.plot(xH, hmean, c='green', label='h mean')

    plt.xlabel('item number')
    plt.legend(loc='upper left')
    plt.savefig(save_name, format='jpg')
    plt.close()

def convertResToJsonStr(res):
    left = res.index('{')
    right = res.index('}')

    return res[left:right+1]


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    os.chdir(submit_path)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')

    print(res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)
    return convertResToJsonStr(res)


if __name__ == '__main__':
    # model_name = './pths_valid/model_epoch_900.pth'
    test_img_path = '/data/home/zjw/dataset/icdar2015/test_images/'
    # test_img_path = './test_images/'
    submit_path = './submit'
    # eval_model(model_name, test_img_path, submit_path)
    model_format = './pths/model_epoch_{}.pth'
    precision = []
    recall = []
    hmean = []
    save_name = './pths/model_test_{}.jpg'
    save_path = './pths/{}.npy'
    for i in range(20, 1001, 20):
        model_name = model_format.format(i)
        res = eval_model(model_name, test_img_path, submit_path)
        resJson = json.loads(res)
        precision.append(resJson['precision'])
        recall.append(resJson['recall'])
        hmean.append(resJson['hmean'])

        # if i%20 == 0:
        #     drawTestAcc('epoch '+str(i), precision, recall, heman, save_name.format(i))
    drawTestAcc('ResNet50 ', precision, recall, hmean, save_name.format('res50'))
    np.save(save_path.format('precision'), np.array(precision, dtype=float))
    np.save(save_path.format('recall'), np.array(recall,dtype=float))
    np.save(save_path.format('hmean'), np.array(hmean, dtype=float))



