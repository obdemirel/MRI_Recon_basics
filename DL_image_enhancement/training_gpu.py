import dataloader as FeedData
import basicResNet as ResNet
import torchvision.models.vgg as models
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
#select the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


## Loss calculation. A mixed l1-l2 norm is used    
def myloss(recon, label):
    return torch.norm(recon-label,p=1)/torch.norm(label,p=1)+torch.norm(recon-label,p=2)/torch.norm(label,p=2)
    #return loss_calc

# training for 300 epoch with LR=1e-3    
def Training(network, device, image_path, epochs=300, batch_size=1, LearningRate=1e-3):

    CartesianData = FeedData.dataloader(datapath = image_path)
    Data_sampler = torch.utils.data.RandomSampler(CartesianData)
    data_loader = torch.utils.data.DataLoader(dataset=CartesianData,
                                               batch_size=batch_size, 
                                               #shuffle=True)  # here we use sampler to test distributed version
                                               sampler = Data_sampler)

    optimizer = optim.Adam(network.parameters(), lr=LearningRate)

    LossFunction = myloss
    # best loss set to inf as starting point
    best_loss = float('inf')
    loss_List = []
    
    
    # epoches
    for epoch in range(epochs):

        network.train()
        loss_buff = []
        # for loop over batches

        for ref_im, acc_im in data_loader:
            optimizer.zero_grad()
            # do zero_grad before every iteration
            
            #Use the followings if you want to use CPUs instead of GPU

            #cc_im = acc_im.to(device=device, dtype=torch.float32)
            #ref_im = ref_im.to(device=device, dtype=torch.float32)
            
            
            acc_im = acc_im.cuda()
            ref_im = ref_im.cuda()

            #print('ref size')
            #print(ref_im.shape,ref_im.dtype)
            #print('input size')
            #print(ref_im.shape,ref_im.dtype)
            # get recon
            recon = network(acc_im)
           
            # get loss
            loss = LossFunction(recon, ref_im)
            print('loss = ',loss.cpu().detach().numpy())

            #print('Max of Recon')
            #print(torch.max(torch.abs(recon_validation)))
            #print('Max of Label')
            #print(torch.max(torch.abs(kspace_validation)))
            #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            #save each epoch if needed, although this may be too much data to save
             

            loss_buff = np.append(loss_buff, loss.item())
            # backpropagate
            loss.backward()
        
            # update parameters
            optimizer.step()
        
    	# only save some of the models
        sio.savemat('LossCurve.mat',{'loss':loss_List})    
        loss_List = np.append(loss_List, np.mean(loss_buff)/2)
        if (epoch % 3 == 0):
            sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            #torch.save(network.state_dict(), 'ModelTemp/best_model_ResNet_Epoch%d.pth'%epoch) 
        if (epoch % 99 == 0):
            #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            torch.save(network.state_dict(), 'ModelTemp/best_model_ResNet_Epoch%d.pth'%epoch) 
        if (epoch % 149 == 0):
            #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            torch.save(network.state_dict(), 'ModelTemp/best_model_ResNet_Epoch%d.pth'%epoch)
        if (epoch % 199 == 0):
            #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            torch.save(network.state_dict(), 'ModelTemp/best_model_ResNet_Epoch%d.pth'%epoch)
        if (epoch % 249 == 0):
            #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            torch.save(network.state_dict(), 'ModelTemp/best_model_ResNet_Epoch%d.pth'%epoch)
        if (epoch % 299 == 0):
            #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy(),'label':ref_im.cpu().detach().numpy()})
            torch.save(network.state_dict(), 'ModelTemp/best_model_ResNet_Epoch%d.pth'%epoch)

        print('$$$$$$$$$$$$$ Average Loss = ', np.mean(loss_buff)/2,', at epoch', epoch,'$$$$$$$$$$$$$$')
        #sio.savemat('ReconEpoch%d.mat'%epoch,{'recon':recon.cpu().detach().numpy()})

            
    return recon, acc_im, ref_im, loss_List
 

if __name__ == "__main__":
    
    rate = 4
    
    # check CUDA availiability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    # Define the ResNet here
    network = ResNet.basicResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)


    # open this of you want to use CPUs only
    #network = network.to(device=device)
    network = network.cuda()
    
    # Image route consist of the database
    imageroute = './database/'
    
    [recon,acc_im,label,loss_List] = Training(network, device, imageroute)
    
    
    # save the last batch 
    sio.savemat('LossCurve.mat',{'loss':loss_List})
