import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os
import lightning
from pytorch_lightning.loggers import TensorBoardLogger

def set_hyperparameters():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Root directory for dataset
    dataroot = "data/mnist"
    # Number of workers for dataloader
    workers = 10
    # Batch size during training
    batch_size = 100
    # Spatial size of training images. All images will be resized to this size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 100
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Number of classes
    opt_class = 10

    return dataroot, workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu, opt_class

def create_dataset(dataroot, image_size):
    train_data = datasets.MNIST(root=dataroot, download=True,
                                train = True,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), 
                                # *Normalize the images, tuple of means and stds, one channal
                            ]))
    test_data = datasets.MNIST(
        root = dataroot,
        train = False,
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    dataset = train_data + test_data
    # from dataprocess import ImageDataset
    # dataroot = "data/comic_faces_dataset"
    # dataset = ImageDataset(dataroot)

    print(f'Total Size of Dataset: {len(dataset)}')

    return dataset

def train():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    dataroot, workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu, opt_class= set_hyperparameters()

    # Create the dataset
    dataset = create_dataset(dataroot, image_size)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=True)

    

    # *Plot some training images
    # real_batch = next(iter(dataloader))
    # print(real_batch)
    # plt.figure(figsize=(10,10))
    # plt.title("Training Images")
    # plt.axis('off')
    # inputs = utils.make_grid(real_batch[:100]*0.5+0.5, nrow=10)
    # plt.imshow(inputs.permute(1, 2, 0))
    # plt.savefig('training_images.png')

    # return 

    # * AutoNet
    from models.Generator import G
    # from models.Discriminator import D
    from models.Discriminator import D2
    
    # * Create the logger
    log_name = f'./logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}-version0'
    if not os.path.exists(log_name):
        os.makedirs(log_name)

    from Condition_AutoNet import AutoNet
    AutoNet = AutoNet(G, D2, nz, nc, ngf, ndf, lr, beta1, log_name, opt_class)


    logger = TensorBoardLogger('tb_logs', name='my_model', max_queue=30)
    Trainer = lightning.Trainer(max_epochs=num_epochs, logger=logger)
    # Trainer = lightning.Trainer(fast_dev_run=True)
    Trainer.fit(AutoNet, dataloader)




if __name__ == '__main__':
    train()