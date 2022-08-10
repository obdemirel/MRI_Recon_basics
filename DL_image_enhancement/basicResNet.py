import torch
import torch.nn as nn
import torch.nn.functional as F
 
class ResidualBlock(nn.Module):
 
    # Unet convs 
    # 2 conv layers, first conv changes dim, the second conv keeps dim
    # notice the activation. there are somebody using leaky relu.
    def __init__(self, input_channels, output_channels, KernelSize=3, para_C=0.1):
        super().__init__() # starting at python 3.x, super(blah, self) can be simplified as super(). 
        self.ConvPart = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2),bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.01, inplace=True)
            nn.Conv2d(output_channels, output_channels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2),bias=False),
            #nn.BatchNorm2d(out_channels),
            #nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.C = para_C
 
 
    def forward(self, x):
        # Residual block scales the convolution parts by C=0.1, added to the input. 
        return x + self.ConvPart(x)*self.C
    



class basicResNet(nn.Module):
    
    def __init__(self, input_channels, intermediateChannels, output_channels, KernelSize=3):
        super().__init__()
        
        # First Layer mapping 2 channels to 64
        self.FirstLayer = nn.Conv2d(input_channels, intermediateChannels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2),bias=False)
        # Then goes through residual blocks, all 64 channels 
        '''
        self.ResNetConv = nn.Sequential(
            ResidualBlock(intermediateChannels,intermediateChannels), # 1 
            nn.Conv2d(intermediateChannels, intermediateChannels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2),bias=False)
            )
        '''
        self.ResNetConv = nn.Sequential(
            ResidualBlock(intermediateChannels,intermediateChannels), # 1 
            ResidualBlock(intermediateChannels,intermediateChannels), # 2 
            ResidualBlock(intermediateChannels,intermediateChannels), # 3
            ResidualBlock(intermediateChannels,intermediateChannels), # 4 
            ResidualBlock(intermediateChannels,intermediateChannels), # 5 
            ResidualBlock(intermediateChannels,intermediateChannels), # 6
            ResidualBlock(intermediateChannels,intermediateChannels), # 7
            ResidualBlock(intermediateChannels,intermediateChannels), # 8
            ResidualBlock(intermediateChannels,intermediateChannels), # 9
            ResidualBlock(intermediateChannels,intermediateChannels), # 10
            ResidualBlock(intermediateChannels,intermediateChannels), # 11
            ResidualBlock(intermediateChannels,intermediateChannels), # 12
            ResidualBlock(intermediateChannels,intermediateChannels), # 13
            ResidualBlock(intermediateChannels,intermediateChannels), # 14
            ResidualBlock(intermediateChannels,intermediateChannels), # 15
            nn.Conv2d(intermediateChannels, intermediateChannels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2),bias=False)
            )
        
        # last layer is just convolution, with 64 to 2 channels as output.
        self.LastLayer = nn.Conv2d(intermediateChannels, output_channels, kernel_size=KernelSize, padding=(KernelSize//2, KernelSize//2),bias=False)
 
    def forward(self, x):
        
        FirstLayerOutput = self.FirstLayer(x)
        return self.LastLayer(FirstLayerOutput + self.ResNetConv(FirstLayerOutput))


# testing, if correct it should give a network description
'''
if __name__ == '__main__':
    net = ZcResNet(input_channels=2, intermediateChannels= 64, output_channels=2)
    print(net)
    
    # check parameters 
    for name, param in net.named_parameters():
        print(name, param.size(), type(param))

'''