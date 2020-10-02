import torch as th
import warnings
import torch
from torch.nn import functional as F

warnings.filterwarnings('ignore')
import torch.nn as nn
EPS = 1e-8

class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,num_embeddings,embedding_dim,beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding_1 = nn.Embedding(self.K, self.D)
        self.embedding_1.weight.data.uniform_(-1 / self.K, 1 / self.K)
        self.embedding_2 = nn.Embedding(self.K, self.D)
        self.embedding_2.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)

        # Compute L2 distance between latents and embedding weights
        dist = th.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               th.sum(self.embedding_1.weight ** 2, dim=1) - \
               2 * th.matmul(flat_latents, self.embedding_1.weight.t())  # [B x K]
        
        # Get the encoding that has the min distance
        encoding_inds = th.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = th.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]

        # Quantize the latents 
        quantized_latents_1 = th.matmul(encoding_one_hot, self.embedding_1.weight)  # [B, D]

        flat_latents_2 = flat_latents - quantized_latents_1

        dist_2 = th.sum(flat_latents_2 ** 2, dim=1, keepdim=True) + \
               th.sum(self.embedding_2.weight ** 2, dim=1) - \
               2 * th.matmul(flat_latents_2, self.embedding_2.weight.t())  # [B x K]
        
        # Get the encoding that has the min distance
        encoding_inds = th.argmin(dist_2, dim=1).unsqueeze(1)  # [B, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = th.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]
        quantized_latents_2 = th.matmul(encoding_one_hot, self.embedding_2.weight)  # [B, D]

        quantized_latents = quantized_latents_1 + quantized_latents_2

        quantized_latents = quantized_latents.view(latents_shape)  # [B x 1 x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        return quantized_latents.contiguous(),vq_loss

class Synthesis(th.nn.Module):
    def __init__(self):
        super(Synthesis, self).__init__()
    def forward(self,excitation,LPC,Initial_state):
        excitation = excitation.squeeze(1)
        B,L = excitation.size()
        B1,L2 = Initial_state.size()

        assert B == B1

        excitation = torch.cat((Initial_state,excitation),1)
        #speech = torch.zeros_like(excitation)
        for i in range(L):
            excitation[:,i+L2] = torch.sum(torch.mul(excitation[:,i:i+L2], LPC),1) + excitation[:,i+L2]
        
        return excitation[:,L2:]

class NNVocode(th.nn.Module):
    def __init__(self,L,B,N,P,H,frame_length):
        """
        L:loop time 
        N:input Channel
        B:hide Channel
        H:output Channel
        P:kernel_size
        """
        super(NNVocode, self).__init__()
        self.L = L
        self.B = B
        self.N =  N
        self.P = P
        self.H = H 
        self.TemporalBlock = TemporalBlock(self.L,self.N,self.B,self.P,self.H)
        self.VQ = VectorQuantizer(4096*2,frame_length)
        self.synthesis = Synthesis()
        self.vq_loss = None 
        self.CONVT = CONVTranspose(self.L,1,self.B,self.P,self.H)

    def forward(self,inSpeech,endcode_speech,LPC,Initial_state):
        inSpeech = inSpeech.unsqueeze(1)
        endcode_speech = endcode_speech.unsqueeze(1)
        y = self.TemporalBlock(inSpeech,endcode_speech)
        y,self.vq_loss = self.VQ(y)
        yy = self.CONVT(y,y)
        speech = self.synthesis(yy,LPC,Initial_state)
        return speech,self.vq_loss

    def  loss_function(self,inSpeech,baseLine):
        origin_power = torch.pow(baseLine, 2).sum(
            1, keepdim=True) + 1e-8  # (batch, 1)

        scale = torch.sum(inSpeech*baseLine, 1, keepdim=True) / \
            origin_power  # (batch, 1)
        
        est_true = scale * baseLine
        est_res = inSpeech - est_true

        true_power = torch.pow(est_true, 2).sum(1) + 1e-8
        res_power = torch.pow(est_res, 2).sum(1) + 1e-8

        return -(10*torch.log10(true_power) - 10*torch.log10(res_power))
        #(X'*X)X/XX = SDR ,
        
class TemporalBlock(th.nn.Module):
    def __init__(self,L,N,B,P,H):
        super(TemporalBlock, self).__init__()
        #B,C,L
        self.layer_norm = ChannelwiseLayerNorm(1)
        self.bottleneck_conv1x1 = nn.Conv1d(1, B, 1, bias=False)
        repeats = []
        for x in range(L):
            dilation = 2**x
            padding = (P-1) * dilation
            repeats +=[DepthAnalysis(B,H,P,padding,dilation)]
        
        self.network = nn.Sequential(*repeats)

    def forward(self,x,y):
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)
        x,y = self.network([x,y])
        return y

class DepthAnalysis(th.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,dilation):
        super(DepthAnalysis, self).__init__()

        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        depthwise_conv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=1, padding=padding,
                                   dilation=dilation, groups=out_channels,
                                   bias=False)
        prelu = nn.LeakyReLU()
        chomp = Chomp1d(padding)
        norm = ChannelwiseLayerNorm(out_channels)

        pointwise_conv = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.net = nn.Sequential(conv1x1,depthwise_conv, prelu,chomp, norm,
                                     pointwise_conv)
        self.y_conv = nn.Conv1d(in_channels, 1, 1, bias=False)
    def forward(self, X):
        x = X[0]
        y = X[1]
        residual = x
        x = residual - self.net(x)
        t = self.y_conv(x)
        _,_,L = y.size()
        _,_,L1 = t.size()
        y = y - t[:,:,L1-L:]
        return [x,y]

class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()

class CONVTranspose(th.nn.Module):
    def __init__(self,L,N,B,P,H):
        super(CONVTranspose,self).__init__()
        self.L = L
        self.B = B
        self.N =  N
        self.P = P
        self.H = H 
        self.layer_norm = ChannelwiseLayerNorm(1)
        self.bottleneck_conv1x1 = nn.ConvTranspose1d(1, B, 1, bias=False)
        repeats = []
        for x in range(L):
            dilation = 2**x
            padding = (P-1) * dilation//2
            repeats +=[Analysis(B,H,P,padding,dilation)]
        self.network = nn.Sequential(*repeats)
    def  forward(self, x,y):
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)
        x,y = self.network([x,y])
        return y

class Analysis(th.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,dilation):
        super(Analysis, self).__init__()
        conv1x1 = nn.ConvTranspose1d(in_channels, out_channels, 1, bias=False)
        depthwise_conv = nn.ConvTranspose1d(out_channels, out_channels, kernel_size,
                                   stride=1, padding=padding,
                                   dilation=dilation, groups=out_channels,
                                   bias=False)
        prelu = nn.LeakyReLU()
        norm = ChannelwiseLayerNorm(out_channels)

        pointwise_conv = nn.ConvTranspose1d(out_channels, in_channels, 1, bias=False)
        self.net = nn.Sequential(conv1x1,depthwise_conv, prelu, norm,
                                     pointwise_conv)
        self.y_conv = nn.ConvTranspose1d(in_channels, 1, 1, bias=False)
    def forward(self, X):
        x = X[0]
        y = X[1]
        residual = x
        x = residual + self.net(x) 
        t = self.y_conv(x)
        y = y + t
        return [x,y]

