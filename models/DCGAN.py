import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        def block(input_dim, output_dim, k, s, p):
            layers = []
            layers.append(nn.ConvTranspose2d(input_dim, output_dim, k, s, p, bias=False))
            layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.block1 = block(latent_dim, 256, 4, 1, 0)
        self.block2 = block(256, 128, 4, 2, 1)
        self.block3 = block(128, 64, 4, 2, 1)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        img = self.block1(z)
        img = self.block2(img)
        img = self.block3(img)
        img = self.final(img)
        
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(input_dim, output_dim, k, s, p):
            layers = []
            layers.append(nn.Conv2d(input_dim, output_dim, k, s, p, bias=False))
            layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.block1 = block(3, 64, 4, 2, 1)
        self.block2 = block(64, 128, 4, 2, 1)
        self.block3 = block(128, 256, 4, 2, 1)
        self.final = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = self.block1(img)
        img = self.block2(img)
        img = self.block3(img)
        output = self.final(img)
        output = output.view(-1, 1)
        return output

if __name__ == "__main__":
    ## Sanity Check
    generator = Generator(100)
    discriminator = Discriminator()
    
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