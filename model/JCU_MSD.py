import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x
  

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal
   


class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, preprocess_config, model_config, train_config):
        super(JCUDiscriminator, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        n_layer = model_config["discriminator"]["n_layer"]
        n_uncond_layer = model_config["discriminator"]["n_uncond_layer"]
        n_cond_layer = model_config["discriminator"]["n_cond_layer"]
        n_channels = model_config["discriminator"]["n_channels"]
        kernel_sizes = model_config["discriminator"]["kernel_sizes"]
        strides = model_config["discriminator"]["strides"]
        self.multi_speaker = model_config["multi_speaker"]

        self.input_projection = LinearNorm(2 * n_mel_channels, 2 * n_mel_channels)
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        )
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(
                LinearNorm(residual_channels, n_channels[n_layer-1]),
            )
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer)
            ]
        )
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, s, t):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x = self.input_projection(
            torch.cat([x_t_prevs, x_ts], dim=-1) #输入concatenate
        ).transpose(1, 2)
        diffusion_step = self.mlp(self.diffusion_embedding(t)).unsqueeze(-1)
        if self.multi_speaker:
            speaker_emb = self.spk_mlp(s).unsqueeze(-1)

        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        x_cond = (x + diffusion_step + speaker_emb) \ #直接相加
            if self.multi_speaker else (x + diffusion_step)
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats




    

class JCU_MSD(nn.Module):
    """ JCU Discriminator """

    def __init__(self,  
                 n_mel_channels = 128,
                 residual_channels = 256,
                 n_layer = 3,
                 n_uncond_layer = 2,
                 n_cond_layer = 2,
                 n_channels = [64, 128, 512, 128, 1],
                 kernel_sizes = [3, 5, 5, 5, 3],
                 strides = [1, 2, 2, 1, 1],
                 multi_speaker = True,
                 external_speaker_dim = 1,
                 encoder_hidden = 256          
                 ):
        super(JCU_MSD, self).__init__()   
        self.multi_speaker = multi_speaker
        self.pool = nn.AvgPool1d(4, 2, 1)
        self.input_projection = LinearNorm(2 * n_mel_channels, 2 * n_mel_channels)
        self.spk_emb = nn.Linear(external_speaker_dim, encoder_hidden)
        
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(
                LinearNorm(residual_channels, n_channels[n_layer-1]),
            )
            
            
            
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer)
            ]
        )
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        # re-write, pay attention to the shape of weights
        self.scale2_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer + n_cond_layer)
            ]
        )
        
        self.scale3_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

            
    def forward(self, inputs, gt, s):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        """
        x = self.input_projection(
            torch.cat([gt, inputs], dim=-1) #输入concatenate
        ).transpose(1, 2)       
        
        if self.multi_speaker:
            speaker_emb = self.spk_emb(s.unsqueeze(-1).float())
            speaker_emb = self.spk_mlp(speaker_emb).unsqueeze(-1)
        
        # scale1
        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        x_cond = (x + speaker_emb) if self.multi_speaker else x
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
           
        # scale2
        x2 = self.input_projection(
            torch.cat([gt, inputs], dim=-1) #输入concatenate
        ).transpose(1, 2)
        
        uncond_feats2 = []
        x2 = self.pool(x2)
        for layer in self.scale2_conv_block:
            x2 = F.leaky_relu(layer(x2), 0.2)
            uncond_feats2.append(x2)


        # scale3
        x3 = self.input_projection(
            torch.cat([gt, inputs], dim=-1) #输入concatenate
        ).transpose(1, 2)
        
        uncond_feats3 = []
        x3 = self.pool(x3)
        x3 = self.pool(x3)
        for layer in self.scale3_conv_block:
            x3 = F.leaky_relu(layer(x3), 0.2)
            uncond_feats3.append(x3)

           
        
        return cond_feats, [uncond_feats, uncond_feats2, uncond_feats3] # feats = [feat1, feat2, feat3, feat4, feat5]
