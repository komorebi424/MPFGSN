import pandas as pd
import torch
from torch import nn
from model.FGSN import fgsn
import numpy as np
import torch.nn.init as init


class mpfgsn(nn.Module):
    def __init__(self, stride, pre_length, embed_size, feature_size, seq_length, hidden_size, patch1_len, patch2_len, patch3_len, patch4_len, patch5_len):
        super(mpfgsn, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.stride = stride
        self.patch1_len = patch1_len
        self.patch2_len = patch2_len
        self.patch3_len = patch3_len
        self.patch4_len = patch4_len
        self.patch5_len = patch5_len
        self.fgsn_model1 = fgsn(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size, self.patch1_len)
        self.fgsn_model2 = fgsn(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size, self.patch2_len)
        self.fgsn_model3 = fgsn(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size, self.patch3_len)
        self.fgsn_model4 = fgsn(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size, self.patch4_len)
        self.fgsn_model5 = fgsn(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size, self.patch5_len)
        # self.fgsn_model4 = fgsn(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size, self.patch4_len)

        self.mlp1 = nn.Parameter(torch.randn(self.patch1_len, 1).double().to('cuda:0').cuda())
        self.mlp2 = nn.Parameter(torch.randn(self.patch2_len, 1).double().to('cuda:0').cuda())
        self.mlp3 = nn.Parameter(torch.randn(self.patch3_len, 1).double().to('cuda:0').cuda())
        self.mlp4 = nn.Parameter(torch.randn(self.patch4_len, 1).double().to('cuda:0').cuda())
        self.mlp5 = nn.Parameter(torch.randn(self.patch5_len, 1).double().to('cuda:0').cuda())

        self.mlpA = nn.Parameter(torch.randn(1024, self.pre_length).double().to('cuda:0').cuda())
        self.mlpB = nn.Parameter(torch.randn(1024, self.pre_length).double().to('cuda:0').cuda())
        # self.mlp4 = nn.Parameter(torch.randn(self.patch4_len, 1).double().to('cuda:0').cuda())
        # self.xp = nn.Parameter(torch.randn(self.patch3_len, 1).double().to('cuda:0').cuda())

        init.xavier_normal_(self.mlp1)
        init.xavier_normal_(self.mlp2)
        init.xavier_normal_(self.mlp3)
        init.xavier_normal_(self.mlp4)
        init.xavier_normal_(self.mlp5)

        init.xavier_normal_(self.mlpA)
        init.xavier_normal_(self.mlpB)
        # init.xavier_normal_(self.mlp4)

        self.fc1 = nn.Linear(1024, 512).double()
        self.relu1 = nn.LeakyReLU(inplace=False)
        self.fc2 = nn.Linear(512, self.hidden_size).double()
        self.relu2 = nn.LeakyReLU(inplace=False)
        self.fc3 = nn.Linear(self.hidden_size, self.pre_length).double()

        self.fc4 = nn.Linear(1024, 1024).double()

        # self.fc2 = nn.Linear(1024, self.pre_length)

        self.cuda()

    def forward(self, x):
        x = x.float()

        self.fc1 = self.fc1.double()
        self.fc4 = self.fc4.double()
        # self.fc3 = self.fc3.double()
        # self.fc2 = self.fc2.double()

        z1 = x
        #print(z1.shape) # [8, 48, 7]
        z1 = z1.permute(0, 2, 1)
        z1 = z1.unfold(dimension=-1, size=self.patch1_len, step=self.patch1_len)
        #print(z1.shape) # [8, 7, 2, 24]
        z1_projected = torch.matmul(z1.double(), self.mlp1)
        #print(z1_projected.shape) # [8, 7, 2, 1]
        z1 = z1_projected.squeeze(3)
        #print(z1.shape) # 8 7 2
        m1 = z1.permute(0, 2, 1)
        F1 = self.fgsn_model1(m1).double().detach().clone()
        Y1 = self.fc1(F1)
        Y1 = self.relu1(Y1.detach().clone())
        Y1 = self.fc2(Y1.detach().clone()).double()
        Y1 = self.relu2(Y1.detach().clone())
        Y1 = self.fc3(Y1.detach().clone())
        G1 = self.fc4(F1)

        z2 = x
        z2 = z2.permute(0, 2, 1)
        z2 = z2.unfold(dimension=-1, size=self.patch2_len, step=self.patch2_len)
        z2_projected = torch.matmul(z2.double(), self.mlp2)
        z2 = z2_projected.squeeze(3)
        m2 = z2.permute(0, 2, 1)
        F2 = self.fgsn_model2(m2).double().detach().clone()
        Y2 = self.fc1((F2 - G1).detach().clone())
        Y2 = self.relu1(Y2.detach().clone())
        Y2 = self.fc2(Y2.detach().clone())
        Y2 = self.relu2(Y2.detach().clone())
        Y2 = self.fc3(Y2.detach().clone())
        G2 = self.fc4((F2 - G1).detach().clone())

        z3 = x
        z3 = z3.permute(0, 2, 1)
        z3 = z3.unfold(dimension=-1, size=self.patch3_len, step=self.patch3_len)
        z3_projected = torch.matmul(z3.double(), self.mlp3)
        z3 = z3_projected.squeeze(3)
        m3 = z3.permute(0, 2, 1)
        F3 = self.fgsn_model3(m3).double().detach().clone()
        Y3 = self.fc1((F3 - G2).detach().clone())
        Y3 = self.relu1(Y3.detach().clone())
        Y3 = self.fc2(Y3.detach().clone())
        Y3 = self.relu2(Y3.detach().clone())
        Y3 = self.fc3(Y3.detach().clone())
        G3 = self.fc4((F3 - G2).detach().clone())

        z4 = x
        z4 = z4.permute(0, 2, 1)
        z4 = z4.unfold(dimension=-1, size=self.patch4_len, step=self.patch4_len)
        z4_projected = torch.matmul(z4.double(), self.mlp4)
        z4 = z4_projected.squeeze(3)
        m4 = z4.permute(0, 2, 1)
        F4 = self.fgsn_model4(m4).double().detach().clone()
        Y4 = self.fc1((F4 - G3).detach().clone())
        Y4 = self.relu1(Y4.detach().clone())
        Y4 = self.fc2(Y4.detach().clone())
        Y4 = self.relu2(Y4.detach().clone())
        Y4 = self.fc3(Y4.detach().clone())
        G4 = self.fc4((F4 - G3).detach().clone())

        z5 = x
        z5 = z5.permute(0, 2, 1)
        z5 = z5.unfold(dimension=-1, size=self.patch5_len, step=self.patch5_len)
        z5_projected = torch.matmul(z5.double(), self.mlp5)
        z5 = z5_projected.squeeze(3)
        m5 = z5.permute(0, 2, 1)
        F5 = self.fgsn_model5(m5).double().detach().clone()
        Y5 = self.fc1((F5 - G4).detach().clone())
        Y5 = self.relu1(Y5.detach().clone())
        Y5 = self.fc2(Y5.detach().clone())
        Y5 = self.relu2(Y5.detach().clone())
        Y5 = self.fc3(Y5.detach().clone())


        return Y1, Y2, Y3, Y4, Y5

