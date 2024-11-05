import torch.nn as nn

class G(nn.Module):
    def __init__(self, nz, ngf, nc, opt_class):
        super(G, self).__init__()
        self.embedding = nn.Embedding(opt_class, nz)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        x = self.embedding(input)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.main(x)
    
if __name__ == '__main__':
    import torch
    nz = 100
    ngf = 64
    nc = 3
    opt_class = 10
    netG = G(nz, ngf, nc, opt_class)
    print(netG)
    input = torch.LongTensor([0,1,2,3,4,5,6,7,8,9])
    print(input.shape)
    o = nn.Embedding(opt_class,nz)(input)
    print(o.shape)
    output = netG(input)
    print(output.size())