import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()

        self.att = nn.Conv2d(
            in_channels=128, out_channels=20, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)
        
        self.bilstm = nn.LSTM(128, 20, batch_first = True, bidirectional = True)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        x = x.transpose(1, 2)
        x = x[:, :, :, None].contiguous()

        att = self.att(x)
        att = torch.sigmoid(att)

        bil, (_,_) = self.bilstm(torch.transpose(x[:, :, :, 0], 1, 2))
        a, b, c = bil.size()
        bil = bil.view(a, b, 2, c//2)
        bil = (bil[:,:,0,:] + bil[:,:,1,:]) / 2
        bil = torch.transpose(bil, 1, 2)
        

        
        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * bil, dim = 2)
        
        x = F.hardtanh(x, 0., 1.)
        return x

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM_Attention, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size = 3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3)
        self.att = nn.Conv2d(32*6*6, 20, kernel_size = 1)
        self.bilstm = nn.LSTM(32*6*6, 20, batch_first = True, bidirectional = True)
        
    def forward(self, x):
        """ CNN
        """
        x = x.transpose(1,3)
        x = x.transpose(2,3)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.transpose(2,3)
        x = x.transpose(1,3)
        a, b, c, d = x.size()
        x = F.dropout(x.view(a, b, c*d))
        
        """attention
        """
        att = torch.sigmoid(self.att(x[:,:,:,None].contiguous()))
        eps = 1e-7
        att = torch.clamp(att, eps, 1 - eps)
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        """BiLSTM
        """
        bil, (_,_) = self.bilstm(x.transpose(1,2))
        a, b, c = bil.size()
        bil = bil.view(a, b, 2, c//2)
        bil = (bil[:,:,0,:] + bil[:,:,1,:]) / 2
        """combining for output
        """
        x = torch.sum(norm_att * bil, dim = 2)
        x = F.hardtanh(x, 0., 1.)
        return x