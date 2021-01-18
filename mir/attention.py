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