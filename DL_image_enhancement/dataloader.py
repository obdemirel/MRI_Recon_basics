
import scipy.io as sio
import glob
from torch.utils.data import Dataset
import os
import numpy as np


class dataloader(Dataset):
    
    def __init__(self, datapath):

        self.data_path = glob.glob(os.path.join(datapath, 'subject*.mat')) 

    def __getitem__(self, index):
        

        reference = np.abs(sio.loadmat(self.data_path[index])['reference'].transpose([0,1]))
        reference = np.expand_dims(reference, axis=0)
        reference = np.concatenate([np.real(reference), np.imag(reference)], axis=0)
        
        accelerated_image = np.abs(sio.loadmat(self.data_path[index])['acc_im'].transpose([0,1]))
        accelerated_image = np.expand_dims(accelerated_image, axis=0)
        accelerated_image = np.concatenate([np.real(accelerated_image), np.imag(accelerated_image)], axis=0)
        '''
        reference = (sio.loadmat(self.data_path[index])['reference'].transpose([0,1]))
        reference = np.expand_dims(reference, axis=0)
        reference = np.concatenate([np.real(reference), np.imag(reference)], axis=0)

        accelerated_image = (sio.loadmat(self.data_path[index])['acc_im'].transpose([0,1]))
        accelerated_image = np.expand_dims(accelerated_image, axis=0)
        accelerated_image = np.concatenate([np.real(accelerated_image), np.imag(accelerated_image)], axis=0)
        '''
        return reference,accelerated_image

        
    def __len__(self):
        return len(self.data_path)
    
