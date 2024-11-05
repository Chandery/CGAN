import torch.nn as nn
import torch

class D(nn.Module):
    def __init__(self, nc, ndf):
        super(D, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # state size. (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class D2(nn.Module):
    def __init__(self, nc, ndf, opt_class):
        super(D2, self).__init__()
        self.label_embedding = nn.Embedding(opt_class, 100)
        self.main = nn.Sequential(
            nn.Linear(64*64+100, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),    # 将输入单元的一部分（本例中为40%）设置为0，有助于 防止过拟合
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        x = torch.cat((input.view(input.size(0), -1), self.label_embedding(label.long())), -1)
        return self.main(x)
    
if __name__ == '__main__':
    input = torch.randn(10, 1, 64, 64)
    label = torch.LongTensor([0,1,2,3,4,5,6,7,8,9])
    x = torch.cat((input.view(input.size(0), -1), nn.Embedding(10,100)(label.long())), -1)
    print(x.shape)
