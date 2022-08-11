
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
#select the GPU you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# count the tunable parameters
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

    
    network.eval()
    # for loop over batches
    
    for ref_im, acc_im in data_loader:
        
        # open these if you want to use CPUs only
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
        
        # Real to complex
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
    



    network = ResNet.basicResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)
    
    #load the latest or best model
    savedModel = torch.load("./ModelTemp/best_model_ResNet_Epoch99.pth")

    # Load historical model
    network.load_state_dict(savedModel)


    network = network.cuda()#network.to(device=device)
   
    imageroute = './database/test/'
    
    # Get EVERYTHING
    output = Testing(network, device, imageroute)
    print(output)
    # Save EVERYTHING
    
    
    