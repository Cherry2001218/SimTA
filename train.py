import torch
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
BATCH_SIZE = 2
SEQ_SIZE = 5
learning_rate = 0.0001
PATH_SAVE = './model/lstm_model.t7'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from geoseg.datasets.LenMyDataset import *

class EncoderMUG2d_LSTM(nn.Module):
    def __init__(self, input_nc=1, encode_dim=1024, lstm_hidden_size=1024, seq_len=SEQ_SIZE, num_lstm_layers=1,
                 bidirectional=False):
        super(EncoderMUG2d_LSTM, self).__init__()
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 32, 4, 2, 1),  # 32*64*64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64*32*32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128*16*16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256*8*8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512*4*4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),  # 512*2*2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024*1*1
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 2048, 4, 2, 1),  # 1024*1*1
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3,padding=1)
        self.pool = nn.BatchNorm2d(1)
       

        self.fc = nn.Linear(1024, encode_dim)
        self.lstm = nn.LSTM(encode_dim, encode_dim, batch_first=True)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * SEQ_SIZE, 3, 256, 256)
        
        x = self.conv1(x)
        x = self.pool(x)
        print(x.shape)
        x = self.encoder(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = x.view(-1, SEQ_SIZE, x.size(1))
        h0, c0 = self.init_hidden(x)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return hn


class DecoderMUG2d(nn.Module):
    def __init__(self, output_nc=1, encode_dim=1024):
        super(DecoderMUG2d, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(encode_dim, 1024 * 1 * 1),
            nn.ReLU(inplace=True)
        )
        self.conv0 = nn.ConvTranspose2d(2048, 1024, 4)  # 512*4*4
        self.bn0 = nn.BatchNorm2d(1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4)  # 512*4*4
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(True),

        self.conv2 = nn.ConvTranspose2d(512, 256, 4)  # 256*10*10
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.ConvTranspose2d(256, 128, 4)  # 128*13*13
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, 4, stride=2)  # 64*28*28
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, 4)  # 32*31*31
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 16, 4, stride=2)  # 16*64*64
        self.bn6 = nn.BatchNorm2d(16)

        self.conv7 = nn.ConvTranspose2d(16, 1, 4, stride=4)  # 3*128*128
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.project(x)
        x = x.view(-1, 2048, 1, 1)
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)
        x = self.conv7(x)
        decode = self.sig(x)
        print(decode.shape)
        return decode


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.n1 = EncoderMUG2d_LSTM()
        self.n2 = DecoderMUG2d()

    def forward(self, x):
        output = self.n1(x)
        output = self.n2(output)
        return output


# 检查梯度消失或梯度爆炸
def check_gradients(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm


if __name__ == '__main__':
    filepath = '/T2007061/liangjiaxvan_workspace/TestDatas'
    train_dataset = LenMyDataset(data_root=filepath, img_dir='TrImage', mask_dir='TrainingLabel',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(256, 256))
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=20,batch_size=BATCH_SIZE)


    model = net()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    #inputs, label = next(iter(train_loader))
    
    for epoch in range(50):
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.
        train_acc = 0.
        for batch in train_loader:
            inputs, label =  batch['img'], batch['gt_semantic_seg']
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            inputs = inputs.to(device)    #1, 通过to（device）方法    	
            inputs = inputs.cuda()		  #2，通过直接指定输入cuda类型	


            label = label.to(device)    #1, 通过to（device）方法    	
            label = label.cuda()		  #2，通过直接指定输入cuda类型	
            output = model(inputs)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.data.cpu().numpy()))
        PATH_SAVE = '/T2007061/liangjiaxvan_workspace/TestDatas/oo.pth'

    torch.save(model.state_dict(), PATH_SAVE)
