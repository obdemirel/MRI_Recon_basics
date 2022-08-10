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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


    
def myloss(recon, label):
    return torch.norm(recon-label,p=1)/torch.norm(label,p=1)+torch.norm(recon-label,p=2)/torch.norm(label,p=2)
    #return loss_calc

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
            
            #acc_im = acc_im.to(device=device, dtype=torch.complex64)
            #ref_im = ref_im.to(device=device, dtype=torch.complex64)
            
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
            # save best weights. Notice it allows loss goes higher but just save yet the best it got
             

            loss_buff = np.append(loss_buff, loss.item())
            # backpropagate
            loss.backward()
        
            # update parameters
            optimizer.step()
        
    	
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
    #device = torch.device('cpu')
    
    
    # Define the unrolled network, using DataParallel to work with multiple GPUs
    network = ResNet.basicResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)

    
    # Load historical model 
    #network.load_state_dict(torch.load("best_model_ResNet_singleData.pth"))
   
    #network = nn.DataParallel(network)
    # move to device
    #network = network.cuda()#network.to(device=device)
    #network = network.to(device=device)
    network = network.cuda()
    
    # lets RO!
    # data_path
    #imageroute = '/home/daedalus2-data2/icarus/Burak_Files/unnormalized_knee/burak_training2/'
    imageroute = '/home/daedalus1-raid1/omer-data/brain_flair/basic_DL_recon/'
    #imageroute = './database/'
    #imageroute = '/home/daedalus1-raid1/omer-data/knee_pdfs_train/' #brain_flair/'

    [recon,acc_im,label,loss_List] = Training(network, device, imageroute)
    #recon = np.squeeze(recon.cpu().detach().numpy())
    #recon = np.squeeze(recon[:,0:1,:,:]+recon[:,1:2,:,:]*1j)
    
    # save the last batch 
    sio.savemat('LossCurve.mat',{'loss':loss_List})
