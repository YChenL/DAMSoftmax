import numpy as np
import os


def data_preparation(data_file):
    ps_list, org_list, spk_list, alpha_list = [], [], [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data_line = line.strip("\n").split()
            ps_list.append(data_line[0])
            org_list.append(data_line[1])
            spk_list.append(data_line[2])
            alpha_list.append(data_line[3])
     
    return ps_list, org_list, spk_list, alpha_list
    
    

def data_prepar(data_file):
    file_list, label_list = [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        # file_list, label_list = [], []
        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分    
            file_list.append(data_line[0])
            label_list.append([data_line[1], data_line[2]])
    
    label_freq_list = []
    for label in label_list:
        if label[0] == 'f0':
            label_freq = 0
        elif label[0] == 'f1':
            label_freq = 1
        elif label[0] == 'f2':
            label_freq = 2
        elif label[0] == 'f3':
            label_freq = 3
        elif label[0] == 'f4':
            label_freq = 4
        elif label[0] == 'f5':
            label_freq = 5
        elif label[0] == 'f6':
            label_freq = 6
        elif label[0] == 'f7':
            label_freq = 7
        elif label[0] == 'f8':
            label_freq = 8
    
        elif label[0] == 'fn1':
            label_freq = -1
        elif label[0] == 'fn2':
            label_freq = -2
        elif label[0] == 'fn3':
            label_freq = -3
        elif label[0] == 'fn4':
            label_freq = -4
        elif label[0] == 'fn5':
            label_freq = -5
        elif label[0] == 'fn6':
            label_freq = -6
        elif label[0] == 'fn7':
            label_freq = -7
        elif label[0] == 'fn8':
            label_freq = -8
        else:
            print('label_error')
        
        label_freq = label_freq + 1e-4*np.random.randn(1)
        label_freq_list.append(label_freq)
    
    
    label_time_list = []
    for label in label_list:
        if label[1] == 't0.5':
            label_time = 0.5
        elif label[1] == 't0.7':
            label_time = 0.7
        elif label[1] == 't0.9':
            label_time = 0.9
        elif label[1] == 't1':
            label_time = 1
        elif label[1] == 't1.1':
            label_time = 1.1
        elif label[1] == 't1.3':
            label_time = 1.3
        elif label[1] == 't1.5':
            label_time = 1.5
        else:
            print('label_error')
       
        label_time = label_time + 1e-3*np.random.randn(1)
        label_time_list.append(label_time)
        
    return file_list, label_freq_list, label_time_list



def data_prepar_spk(data_file):
    file_list, label_list, xvector_list = [], [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        # file_list, label_list = [], []
        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分    
            file_list.append(data_line[0])
            xvector_list.append(data_line[1])
            label_list.append([data_line[2], data_line[3]])
    
    label_freq_list = []
    for label in label_list:
        if label[0] == 'f0':
            label_freq = 0
        elif label[0] == 'f1':
            label_freq = 1
        elif label[0] == 'f2':
            label_freq = 2
        elif label[0] == 'f3':
            label_freq = 3
        elif label[0] == 'f4':
            label_freq = 4
        elif label[0] == 'f5':
            label_freq = 5
        elif label[0] == 'f6':
            label_freq = 6
        elif label[0] == 'f7':
            label_freq = 7
        elif label[0] == 'f8':
            label_freq = 8
    
        elif label[0] == 'fn1':
            label_freq = -1
        elif label[0] == 'fn2':
            label_freq = -2
        elif label[0] == 'fn3':
            label_freq = -3
        elif label[0] == 'fn4':
            label_freq = -4
        elif label[0] == 'fn5':
            label_freq = -5
        elif label[0] == 'fn6':
            label_freq = -6
        elif label[0] == 'fn7':
            label_freq = -7
        elif label[0] == 'fn8':
            label_freq = -8
        else:
            print('label_error')
        
        label_freq = label_freq + 1e-3*np.random.randn(1)
        label_freq_list.append(label_freq)
    
    
    label_time_list = []
    for label in label_list:
        if label[1] == 't0.4':
            label_time = 0.4
        elif label[1] == 't0.6000000000000001':
            label_time = 0.6
        elif label[1] == 't0.8':
            label_time = 0.8
        elif label[1] == 't1.0':
            label_time = 1
        ########################        
        elif label[1] == 't1.wav':
            label_time = 1
        ########################   
        elif label[1] == 't1.2':
            label_time = 1.2
        elif label[1] == 't1.4':
            label_time = 1.4
        elif label[1] == 't1.5999999999999999':
            label_time = 1.6
        elif label[1] == 't1.7999999999999998':
            label_time = 1.8
        else:
            print('label_error:', label[1])
       
        label_time = label_time + 1e-4*np.random.randn(1)
        label_time_list.append(label_time)
        
    return file_list, xvector_list, label_freq_list, label_time_list



def data_prepar2(data_file):
    file_list, label_list = [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        # file_list, label_list = [], []
        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
            if data_line[2]=='t1.7999999999999998':
                pass
            else:
                file_list.append(data_line[0])
                label_list.append([data_line[1], data_line[2]])
    
    label_freq_list = []
    for label in label_list:
        if label[0] == 'f0':
            label_freq = 0
        elif label[0] == 'f1':
            label_freq = 1
        elif label[0] == 'f2':
            label_freq = 2
        elif label[0] == 'f3':
            label_freq = 3
        elif label[0] == 'f4':
            label_freq = 4
        elif label[0] == 'f5':
            label_freq = 5
        elif label[0] == 'f6':
            label_freq = 6
        elif label[0] == 'f7':
            label_freq = 7
        elif label[0] == 'f8':
            label_freq = 8
    
        elif label[0] == 'fn1':
            label_freq = -1
        elif label[0] == 'fn2':
            label_freq = -2
        elif label[0] == 'fn3':
            label_freq = -3
        elif label[0] == 'fn4':
            label_freq = -4
        elif label[0] == 'fn5':
            label_freq = -5
        elif label[0] == 'fn6':
            label_freq = -6
        elif label[0] == 'fn7':
            label_freq = -7
        elif label[0] == 'fn8':
            label_freq = -8
        else:
            print('label_error')
        
        label_freq = label_freq + 1e-4*np.random.randn(1)
        label_freq_list.append(label_freq)
    
    
    label_time_list = []
    for label in label_list:
        if label[1] == 't0.4':
            label_time = 0.4
        elif label[1] == 't0.6000000000000001':
            label_time = 0.6
        elif label[1] == 't0.8':
            label_time = 0.8
        elif label[1] == 't1.0':
            label_time = 1
        ########################        
        elif label[1] == 't1.wav':
            label_time = 1
        ########################   
        elif label[1] == 't1.2':
            label_time = 1.2
        elif label[1] == 't1.4':
            label_time = 1.4
        elif label[1] == 't1.5999999999999999':
            label_time = 1.6
        else:
            print('label_error:', label[1])
       
        label_time = label_time + 1e-4*np.random.randn(1)
        label_time_list.append(label_time)
        
    return file_list, label_freq_list, label_time_list



def data_prepar_joint(data_file):
    file_list, label_list = [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        # file_list, label_list = [], []
        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分    
            file_list.append(data_line[0])
            label_list.append(float(data_line[1]))
   
    return file_list, label_list


def data_prepar_joint_spk(data_file):
    file_list, file_org_list,label_list = [], [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        # file_list, label_list = [], []
        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分    
            file_list.append(data_line[0])
            file_org_list.append(data_line[1])
            label_list.append(float(data_line[2]))
   
    return file_list, file_org_list, label_list


def data_prepar_audio(data_file):
    file_list, file_org_list,label_list = [], [], []
    with open(data_file, 'r', encoding='utf-8') as infile:
        # file_list, label_list = [], []
        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分    
            file_list.append(data_line[0])
            file_org_list.append(data_line[1])
   
    return file_list, file_org_list