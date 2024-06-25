import torch
import torch.nn as nn
import torch.nn.functional as F



class fgsn(nn.Module):
    def __init__(self, pre_length, embed_size, seq_length,
                 feature_size, hidden_size, patch_len, hard_thresholding_fraction=1, hidden_size_factor=1,
                 sparsity_threshold=0.01):
        super(fgsn, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.patch_len = patch_len
        self.patch_num = self.seq_length // self.patch_len
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))  # embed_size=128
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.patch_num, 8).double())
        self.to('cuda:0')

    def token_embedding(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return torch.mul(x, y)

    def fourierGC(self, x, B, N, L):  # 8 8 128
        w10 = torch.diag(self.w1[0])
        w11 = torch.diag(self.w1[1])
        w20 = torch.diag(self.w2[0])
        w21 = torch.diag(self.w2[1])
        w30 = torch.diag(self.w3[0])
        w31 = torch.diag(self.w3[1])
        x = x.permute(0, 2, 1)  # 8 128 8
        self.fc_real = nn.Linear((N * L) // 2 + 1, (N * L) // 5).to('cuda:0').double()
        self.fc_imag = nn.Linear((N * L) // 2 + 1, (N * L) // 5).to('cuda:0').double()
        self.ba = nn.Parameter(self.scale * torch.randn(2, (N * L) // 5)).to('cuda:0')
        x_real_transformed = self.fc_real(x.real)
        x_imag_transformed = self.fc_imag(x.imag)
        oa_real = F.relu(
            x_real_transformed - \
            x_imag_transformed + \
            self.ba[0]
        )
        oa_imag = F.relu(
            x_imag_transformed + \
            x_real_transformed + \
            self.ba[1]
        )
        a = torch.stack([oa_real, oa_imag], dim=-1)
        a = F.softshrink(a, lambd=self.sparsity_threshold)  # 8 128 8 2  8 128 7 2
        a = a.permute(0, 2, 1, 3)  # 8 8 128 2  8 7 128 2
        oa_real = oa_real.permute(0, 2, 1)
        oa_imag = oa_imag.permute(0, 2, 1)
        o1_real = F.relu(
            torch.einsum('bli,i->bli', oa_real, w10) - \
            torch.einsum('bli,i->bli', oa_imag, w11) + \
            self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum('bli,i->bli', oa_imag, w10) + \
            torch.einsum('bli,i->bli', oa_real, w11) + \
            self.b1[1]
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = y + a
        o2_real = F.relu(
            torch.einsum('bli,i->bli', o1_real, w20) - \
            torch.einsum('bli,i->bli', o1_imag, w21) + \
            self.b2[0]
        )
        o2_imag = F.relu(
            torch.einsum('bli,i->bli', o1_imag, w20) + \
            torch.einsum('bli,i->bli', o1_real, w21) + \
            self.b2[1]
        )
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y
        o3_real = F.relu(
            torch.einsum('bli,i->bli', o2_real, w30) - \
            torch.einsum('bli,i->bli', o2_imag, w31) + \
            self.b3[0]
        )
        o3_imag = F.relu(
            torch.einsum('bli,i->bli', o2_imag, w30) + \
            torch.einsum('bli,i->bli', o2_real, w31) + \
            self.b3[1]
        )
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x  # 8 7 128 2
        z = z.permute(0, 2, 1, 3)
        o3_real = o3_real.permute(0, 2, 1)  # 8 128 7
        o3_imag = o3_imag.permute(0, 2, 1)  # 8 128 7

        self.fc_real1 = nn.Linear((N * L) // 5, (N * L) // 2 + 1).to('cuda:0').double()
        self.fc_imag1 = nn.Linear((N * L) // 5, (N * L) // 2 + 1).to('cuda:0').double()
        self.bb = nn.Parameter(self.scale * torch.randn(2, (N * L) // 2 + 1)).to('cuda:0')
        x_real_transformed1 = self.fc_real1(o3_real)
        x_imag_transformed1 = self.fc_imag1(o3_imag)
        ob_real = F.relu(
            x_real_transformed1 - \
            x_imag_transformed1 + \
            self.bb[0]
        )
        ob_imag = F.relu(
            x_imag_transformed1 + \
            x_real_transformed1 + \
            self.bb[1]
        )
        b = torch.stack([ob_real, ob_imag], dim=-1)  # 8 128 8 2
        b = F.softshrink(b, lambd=self.sparsity_threshold)

        b = torch.view_as_complex(b)
        b = b.permute(0, 2, 1)  # 8 8 128
        return b

    def forward(self, x):  # 4 96 7
        x = x.permute(0, 2, 1).contiguous()  # old 4 7 96
        B, N, L = x.shape  # old 4 7 96
        x = x.reshape(B, -1)  # old 4 672
        x = self.token_embedding(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # 4 337 128
        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)  # old 4 337 128
        bias = x  # old 4 337 128
        x = self.fourierGC(x, B, N, L)  # old 4 337 128
        x = x + bias  # old 4 337 128
        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)  # old 4 337 128
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")  # old 4 672 128
        x = x.reshape(B, N, L, self.embed_size)  # old 4 7 96 128
        x = x.permute(0, 1, 3, 2)  # B, N, D, L 4 7 128 96
        # projection
        x = x.double()
        self.embeddings_10 = nn.Parameter(self.embeddings_10.double())
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)  # old 4 7 1024
        x = x.float()
        return x