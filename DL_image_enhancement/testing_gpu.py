
import dataloader as FeedData
import basicResNet as ResNet

from torch import optim
import torch.nn as nn
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import argparse
import torch.distributed as dist
import torch.utils.data.distributed
from collections import OrderedDict
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable_num = ',trainable_num)
    print('Total var num = ',total_num)
    return 1

def Testing(network, device, image_path):

    Datain = FeedData.dataloader(datapath = image_path)
    Data_sampler = torch.utils.data.RandomSampler(Datain)
    data_loader = torch.utils.data.DataLoader(dataset=Datain,
                                               batch_size=1, 
                                               #shuffle=True)  # here we use sampler to test distributed version
                                               sampler = Data_sampler)

    # epoches

    # start torch.nn.module's training mode
    network.eval()
    # for loop over batches
    
    for ref_im, acc_im in data_loader:
        
        # get the data from data loader. 
        #acc_im = acc_im.to(device=device, dtype=torch.float32)
        #ref_im = ref_im.to(device=device, dtype=torch.float32)
        
        acc_im = acc_im.cuda()
        ref_im = ref_im.cuda()
        
        # get recon
        time_s = time.time()
        recon = network(acc_im)
        
        time_e = time.time()
        recon = recon.cpu().detach().numpy()
        acc_im = acc_im.cpu().detach().numpy()
        ref_im = ref_im.cpu().detach().numpy()
        # R2C
        recon = np.squeeze(recon[:,0,:,:]+recon[:,1,:,:]*1j).transpose([0,1])
        ref_im = np.squeeze(ref_im[:,0,:,:]+ref_im[:,1,:,:]*1j).transpose([0,1])
        acc_im = np.squeeze(acc_im[:,0,:,:]+acc_im[:,1,:,:]*1j).transpose([0,1])
        
        print('Testing file_ time cost = ',time_e-time_s,' secs, including saving intermediate results')
        sio.savemat(('./test/'+'test.mat'),{'recon':recon, 'label':ref_im,'input_im':acc_im})
    for i in network.state_dict():
        print(i) 
    get_parameter_number(network)   
    return 'testing done'
 

if __name__ == "__main__":
    
    
    # check CUDA availiability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    


    # Define the unrolled network, using DataParallel to work with multiple GPUs
    network = ResNet.basicResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)

    savedModel = torch.load("./ModelTemp/best_model_ResNet_Epoch99.pth")

    # Load historical model
    #network.load_state_dict(new_state_dict)
    network.load_state_dict(savedModel)

    #network = nn.DataParallel(network)
    # move to device
    network = network.cuda()#network.to(device=device)
    #network = network.to(device=device)
    
    # lets RO!
    
    # data_path
    #imageroute = '/home/daedalus2-data2/icarus/Burak_Files/unnormalized_knee/chi_test/yedek2/'
    #imageroute = '/home/daedalus1-raid1/omer-data/brain_flair/test/noPF/'
    #imageroute = './database/'
    imageroute = '/home/daedalus1-raid1/omer-data/brain_flair/basic_DL_recon/test/'
    
    # Get EVERYTHING
    output = Testing(network, device, imageroute)
    print(output)
    # Save EVERYTHING
    
    
    