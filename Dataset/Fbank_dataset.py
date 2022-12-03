# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import torchaudio
# We're using the audio processing from TacoTron2 to make sure it matches
from model.layers import TacotronSTFT
from scipy.io.wavfile import read


MAX_WAV_VALUE = 32768.0


class Fbank_dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_file, valid, segment_length, filter_length, n_mel_channels,
                 hop_length, win_length, mel_fmin, mel_fmax, target_length=500, sampling_rate=22050):  
        super(Fbank_dataset, self).__init__()
        self.filter_length  = filter_length
        self.hop_length     = hop_length
        self.win_length     = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin       = mel_fmin
        self.mel_fmax       = mel_fmax  
        self.valid          = valid
        self.segment_length = segment_length
        self.target_length  = target_length
        self.sampling_rate  = sampling_rate
        
        self.ps_files, self.org_files, self.spkid, self.alpha = self.data_preparation(data_file)   
        random.seed(1234)
        random.shuffle(self.ps_files)  
        random.seed(1234)
        random.shuffle(self.org_files)  
        random.seed(1234)
        random.shuffle(self.spkid)  
        random.seed(1234)
        random.shuffle(self.alpha)
   

        self.stft = TacotronSTFT(filter_length= filter_length,
                                 hop_length   = hop_length,
                                 win_length   = win_length,
                                 sampling_rate= sampling_rate,
                                 mel_fmin     = mel_fmin, 
                                 mel_fmax     = mel_fmax)
   

    def data_preparation(self, data_file):
        ps_list, org_list, spk_list, alpha_list = [], [], [], []
        with open(data_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                data_line = line.strip("\n").split()
                ps_list.append(data_line[0])
                org_list.append(data_line[1])
                spk_list.append(data_line[2])
                alpha_list.append(data_line[3])  
        return ps_list, org_list, spk_list, alpha_list

    
    def files_to_list(self, data_path):
        """
        Load all .wav files in data_path
        """
        files = [os.path.join(data_path, f.rstrip()) for f in os.listdir(data_path) if len(f)>=4 and f[-4:]=='.wav']
        return files

    
    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        sampling_rate, data = read(full_path)
        return torch.from_numpy(data).float(), sampling_rate

    
    def get_mel(self, audio, sr):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).T     
        # #cut and pad
        target_length = self.target_length
        n_frames = melspec.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            melspec = m(melspec)
        else:
            melspec = melspec[0:target_length, :]   
        return melspec
    

    def __getitem__(self, index):
        # Read audio
        ps_filename       = self.ps_files[index]
        org_filename      = self.org_files[index] 
        ps_audio , sr     = self.load_wav_to_torch(ps_filename )
        org_audio, org_sr = self.load_wav_to_torch(org_filename)
        
        if sr != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sr, self.sampling_rate))

        ps_mel  = self.get_mel(ps_audio , sr)
        org_mel = self.get_mel(org_audio, org_sr)
        spk_id  = self.spkid[index]
        alpha   = self.alpha[index]

        return (ps_mel, org_mel, spk_id, alpha)
            

    def __len__(self):
        return len(self.ps_files)

