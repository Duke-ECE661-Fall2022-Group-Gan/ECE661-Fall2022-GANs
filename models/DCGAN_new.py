import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


# Following table 2, AC-GAN paper
class Generator(nn.Module):
    def __init__(self, latent_dim):
        '''
        input: random sampled latent vector z (B x 110)
        linear: (B x 384)
        reshape: (B x 384 x 1 x 1)
        block1: (B x 192 x 4 x 4)
        block2: (B x 96 x 8 x 8)
        block3: (B x 48 x 16 x 16)
        final: (B x 3 x 32 x 32)
        '''
        super(Generator, self).__init__()

        def block(input_dim, output_dim, k, s, p):
            layers = []
            layers.append(nn.ConvTranspose2d(input_dim, output_dim, k, s, p, bias=False))
            layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.linear = nn.Linear(latent_dim, 384)

        self.block1 = block(384, 192, 4, 1, 0)
        self.block2 = block(192, 96, 4, 2, 1)
        self.block3 = block(96, 48, 4, 2, 1)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.linear(z)
        z = z.view(z.size(0), 384, 1, 1)
        img = self.block1(z)
        img = self.block2(img)
        img = self.block3(img)
        img = self.final(img)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        '''
        input: image (B x 3 x 32 x 32)
        block1: (B x 16 x 16 x 16)
        block2: (B x 32 x 16 x 16)
        block3: (B x 64 x 8 x 8)
        block4: (B x 128 x 8 x 8)
        block3: (B x 256 x 4 x 4)
        block4: (B x 512 x 4 x 4)
        final: (B x 11 x 1 x 1)
        output: first feature as fake/real
                last 10 features as class label
        '''
        super(Discriminator, self).__init__()

        def block(input_dim, output_dim, k, s, p, bn=True):
            layers = []
            layers.append(nn.Conv2d(input_dim, output_dim, k, s, p, bias=False))
            if bn: layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout())
            return nn.Sequential(*layers)

        self.block1 = block(3, 16, 3, 2, 1, False)
        self.block2 = block(16, 32, 3, 1, 1)
        self.block3 = block(32, 64, 3, 2, 1, False)
        self.block4 = block(64, 128, 3, 1, 1)
        self.block5 = block(128, 256, 3, 2, 1, False)
        self.block6 = block(256, 512, 3, 1, 1)
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, img):
        img = self.block1(img)
        img = self.block2(img)
        img = self.block3(img)
        img = self.block4(img)
        img = self.block5(img)
        img = self.block6(img)
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
    if device == 'cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    # Generator Definition
    generator = generator.to(device)
    summary(generator, (100,))

    # Test forward pass
    z = torch.randn(5, 100)
    z = z.to(device)
    out = generator.forward(z)
    # Check output shape
    assert (out.detach().cpu().numpy().shape == (5, 3, 32, 32))
    print("Forward pass successful")

    # sanity check for the correctness of Discriminator
    # Discriminator Definition
    discriminator = discriminator.to(device)
    summary(discriminator, (3, 32, 32))
    # Test forward pass
    data = torch.randn(5, 3, 32, 32)
    data = data.to(device)
    out = discriminator.forward(data)
    # Check output shape
    assert (out.detach().cpu().numpy().shape == (5, 1))
    print("Forward pass successful")