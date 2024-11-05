import lightning
import torch
from torchvision import utils, datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import time

class AutoNet(lightning.LightningModule):
    def __init__(self, Generator, Discriminator, nz, nc, ngf, ndf, lr, beta1, log_name, opt_class):
        super(AutoNet, self).__init__()
        self.automatic_optimization = False

        # * Define the generator and discriminator
        self.G = Generator(nz, ngf, nc, opt_class)
        self.D = Discriminator(nc, ndf, opt_class)
        
        # * Define the hyperparameters & device & labels
        self.lr = lr
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.beta = beta1

        self.real_label = 1
        self.fake_label = 0

        self.epoch = -1

        # * initialize the weights
        self.G.apply(self.weight_init)
        self.D.apply(self.weight_init)

        self.criterion = nn.BCELoss()

        # * define logger
        logging.basicConfig(filename=log_name+"/train_log.log", level=logging.INFO)


    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # * update the state_dict because of two models
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
        })
        return state_dict
    
    # def load_state_dict(self, state_dict, strict=True):
    #     self.G.load_state_dict(state_dict.pop('G'))
    #     self.D.load_state_dict(state_dict.pop('D'))
    #     super().load_state_dict(state_dict, strict)
    #     return self

    def training_step(self, batch, batch_idx):
        
        opt_G, opt_D = self.optimizers()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        real, cond= batch
        b_size = real.size(0)
        label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
        output = self.D(real, cond).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = self.make_condition_noise(cond)
        # noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        fake = self.G(noise)
        label.fill_(0)
        output = self.D(fake.detach(), cond).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        opt_D.step()
        opt_D.zero_grad()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        label.fill_(self.real_label)
        output = self.D(fake, cond).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        opt_G.step()
        opt_G.zero_grad()

        # * saving & logging
        self.errG = errG
        self.errD = errD
        self.D_x = D_x
        self.D_G_z1 = D_G_z1
        self.D_G_z2 = D_G_z2

        self.log('errD', errD)
        self.log('D_x', D_x)
        self.log('errG', errG)
        # self.logger.experiment.add_scalars('D(G(z))', {'D(G(z))_1': D_G_z1, 'D(G(z))_2': D_G_z2}, self.current_epoch)

    def make_condition_noise(self, cond):
        # num_classes = cond.max().item() + 1  # 假设cond中的标签是从0开始的
        # noise_1 = torch.randn(batch_size, nz-num_classes, 1, 1, device=self.device)
        # noise_2 = F.one_hot(cond, num_classes).float().unsqueeze(2).unsqueeze(3).to(self.device)
        # noise = torch.cat((noise_1, noise_2), 1)
        # print(noise.shape)
        noise = cond.long()
        # print(noise)
        # print(noise.shape)
        return noise

    # def validation_step(self, *args: torch.Any, **kwargs: torch.Any) -> torch.Tensor | os.Mapping[str, torch.Any] | None:
    #     if self.current_epoch % 5 == 0:
    #         fake_images = []
    #         for i in range(100):
    #             noise = self.make_condition_noise(torch.tensor([i % 10]))
    #             fake_image = self.G(noise).squeeze()
    #             fake_images.append(fake_image)
    #         fake_images_tensor = torch.stack(fake_images)
    #         # print(fake_images_tensor.shape)
    #         if len(fake_images_tensor.shape) == 3:
    #             fake_images_tensor = fake_images_tensor[:, None, :, :]

    #         grid = utils.make_grid(fake_images_tensor, nrow=10)
    #         # print("start logging")
    #         self.logger.experiment.add_image('Generated Images', grid, self.current_epoch)
    #         # print("end logging")
    #     return super().validation_step(*args, **kwargs)

    def on_train_epoch_end(self):
        if self.current_epoch != self.epoch:
            self.epoch = self.current_epoch
            # print(self.epoch)
            logging.info(
                f'Epoch: {self.current_epoch}, \n'
                f'Loss-D: {self.errD.item():.4f}, \n'
                f'Loss-G: {self.errG.item():.4f}, \n'
                f'D(x): {self.D_x:.4f}, \n'
                f'D(G(z)): [{self.D_G_z1:.4f}/{self.D_G_z2:.4f}]\n'
            )
            # *save the model
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')

            checkpoint_path = f'./checkpoints/epoch_{self.current_epoch}.ckpt'
            torch.save(self.state_dict(), checkpoint_path)

            # 训练完生成一轮
            if self.current_epoch % 10 == 0:
                fake_images = []
                for i in range(100):
                    noise = self.make_condition_noise(torch.tensor([i % 10])).to(self.device)
                    fake_image = self.G(noise).squeeze()
                    fake_images.append(fake_image)
                fake_images_tensor = torch.stack(fake_images)
                # print(fake_images_tensor.shape)
                if len(fake_images_tensor.shape) == 3:
                    fake_images_tensor = fake_images_tensor[:, None, :, :]

                grid = utils.make_grid(fake_images_tensor, nrow=10)
                # print("start logging")
                self.logger.experiment.add_image('Generated Images', grid, self.current_epoch)
                # print("end logging")
        



    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta, 0.999))

        return opt_G, opt_D

if __name__=="__main__":
    cond = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    num_classes = cond.max().item() + 1
    noise_2 = F.one_hot(cond, num_classes).float().unsqueeze(2).unsqueeze(3)
    print(noise_2.shape)

