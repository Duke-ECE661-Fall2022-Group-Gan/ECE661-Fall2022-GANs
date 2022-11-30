import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class Generator(nn.Module):
    def __init__(self,
                    latent_dim = 100,
                    img_size = 32,
                    n_channels = 3):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.n_channels = n_channels
        
        def block(input_dim, output_dim, normalize = True):
            layers = []
            layers.append(nn.Linear(input_dim, output_dim))
            if normalize: layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.block1 = block(latent_dim, 128, normalize = False)
        self.block2 = block(128, 256)
        self.block3 = block(256, 512)
        self.block4 = block(512, 1024)
        self.final = nn.Sequential(
            nn.Linear(1024, self.n_channels * self.img_size * self.img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.block1(z)
        img = self.block2(img)
        img = self.block3(img)
        img = self.block4(img)
        img = self.final(img)
        img = img.view(img.size(0), self.n_channels, self.img_size, self.img_size)
        return img
    
class Discriminator(nn.Module):
    def __init__(self,
                    img_size = 32,
                    n_channels = 3):
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.n_channels = n_channels
        
        def block(input_dim, output_dim):
            layers = []
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.block1 = block(self.n_channels * self.img_size * self.img_size, 1024)
        self.block2 = block(1024, 512)
        self.block3 = block(512, 256)
        self.block4 = block(256, 128)
        self.final = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        img = self.block1(img)
        img = self.block2(img)
        img = self.block3(img)
        img = self.block4(img)
        output = self.final(img)
        return output

if __name__ == "__main__":            
    ## Sanity Check
    generator = Generator(100,32,3)
    discriminator = Discriminator(32,3)
    
    # sanity check for the correctness of Generator
    # GPU check                
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    # Generator Definition  
    generator = generator.to(device)
    summary(generator, (100,))

    # Test forward pass
    z = torch.randn(5,100)
    z = z.to(device)
    out = generator.forward(z)
    # Check output shape
    assert(out.detach().cpu().numpy().shape == (5, 3, 32, 32))
    print("Forward pass successful")
    
    # sanity check for the correctness of Discriminator
    # Discriminator Definition  
    discriminator = discriminator.to(device)
    summary(discriminator, (3, 32, 32))
    # Test forward pass
    data = torch.randn(5,3,32,32)
    data = data.to(device)
    out = discriminator.forward(data)
    # Check output shape
    assert(out.detach().cpu().numpy().shape == (5,1))
    print("Forward pass successful")