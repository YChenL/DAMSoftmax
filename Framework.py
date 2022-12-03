import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from model.layers import TacotronSTFT
from model.ast_model import ASTModel
from model.wavenet import WaveNet
from model.JCU_MSD import JCU_MSD
from model.losses import DAMSoftmax, PredictionLoss, generator_loss, \
                         discriminator_loss, get_mel_loss, get_fm_loss


class ASVframework(nn.Module):
    def __init__(self,            
                 input_tdim = 500, 
                 emb_dim = 768,
                 class_nums = 5999,
                 hidden_nums = 1000
                ):
        
        super(ASVframework, self).__init__()
        # FT
        self.target_lenght = input_tdim
        self.stft = TacotronSTFT(filter_length=1024,
                                 hop_length=256,
                                 win_length=1024,
                                 sampling_rate=22050,
                                 mel_fmin=0, mel_fmax=8000)      
        
        # ASV   
        self.encoder = ASTModel(label_dim=1, fshape=128, tshape=2, fstride=128, tstride=1,
                       input_fdim=128, input_tdim=input_tdim, model_size='base',
                       pretrain_stage=False, load_pretrained_mdl_path='save_model/SSAST-Base-Frame-400.pth').cuda()
        self.closs = DAMSoftmax(emb_dim, class_nums).cuda()
        
        self.ploss = PredictionLoss(emb_dim, hidden_nums).cuda()
        
        self.asv_opt  = optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-4},
                                   {'params': self.closs.parameters()}], lr=1e-3, betas=(0.9, 0.99))
        self.pred_opt = optim.Adam(self.ploss.parameters(), lr=1e-4, betas=(0.9, 0.99))
        
        # FRN 
        self.gen = WaveNet(gin_channels=1, upsample_conditional_features=True).cuda()
        self.JCUMSD = JCU_MSD().cuda()
        self.genloss = generator_loss
        self.disloss = discriminator_loss
        self.specloss = get_mel_loss
        self.fmloss = get_fm_loss
        self.g_opt = optim.Adam(self.gen.parameters(), lr=1e-4, betas=(0.9, 0.99))
        self.d_opt = optim.Adam(self.JCUMSD.parameters(), lr=1e-4, betas=(0.9, 0.99)) 

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        sampling_rate, data = read(full_path)
        return torch.from_numpy(data).float(), sampling_rate

    def get_mel(self, audio, sr):
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).T
        '''
         mel = (D, T)
        '''
        #cut and pad
        n_frames = melspec.shape[0]
        p = self.target_lenght - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            melspec = m(melspec)
        elif p < 0:
            melspec = melspec[0:target_length, :]       
            
        return melspec
           
    def model_update(params, step, loss, optimizer):
        grad_acc_step = 1
        grad_clip_thresh = 1
        # Backward
        loss = (loss / grad_acc_step).backward()
        if step % grad_acc_step == 0:
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(params, grad_clip_thresh)

            # Update weights
            optimizer.step()
            optimizer.zero_grad()    
        
    def train_step(self, fbank_s, fbank_o, label, a, step):
        '''
         inputs: fbanks of pitch-shifted voice
        '''
        # train ASV system
        spk_emb = self.encoder(fbank_s, task='ft_emb')
        cls_loss, acc = self.closs(spk_emb, label)    
        model_update([{'params': self.encoder.parameters()},
                      {'params': self.closs.parameters()}],
                     step, cls_loss, self.asv_opt) 
        
        pred_loss, pred_a = self.ploss(spk_emb.detach(), a)
        model_update(self.ploss.parameters(),
                     step, pred_loss, self.pred_opt) 
        
        # train FRN
        fbank_recon = self.gen(fbank_s.transpose(2,1), g=-pred_a).transpose(2,1)
        # train D
        D_real_cond, D_real_uncond = D(fbank_o)
        D_fake_cond, D_fake_uncond = D(fbank_recon.detach())
        
        loss_D = self.disloss(D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond)
        model_update(self.JCUMSD.parameters(), 
                     step, loss_D, self.d_opt)    
            
        # train G
        fbank_recon = self.gen(fbank_s.transpose(2,1), g=-pred_a).transpose(2,1)
        D_real_cond, D_real_uncond = D(fbank_o)
        D_fake_cond, D_fake_uncond = D(fbank_recon.detach())
        
        loss_adv   = self.genloss(D_fake_cond, D_fake_uncond) 
        loss_recon = self.specloss(fbank_recon, fbank_o)
        loss_FM    = self.fmloss(D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond)
        lambda_FM = loss_recon.item() / loss_FM.item()     
        loss_total = loss_adv + loss_recon + lambda_FM*loss_FM     
        model_update(self.gen.parameters(), 
                     step, loss_total, self.g_opt)        
        
        # viusalizations
        
        
        ...
        return acc
        
        
    def fit(self, train_loader, test_loader, EPOCHS):
        for epoch in range(EPOCHS):
            for i, (fbank_s, fbank_o, label, a) in enumerate(train_loader):
                fbank_s = fbank_s.cuda()
                fbank_o = fbank_o.cuda()
                label = label.cuda() 
                a = a.cuda()
                
                self.train_step(fbank_s, fbank_o, label, a, i)
            
    

        