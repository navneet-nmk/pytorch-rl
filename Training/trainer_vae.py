"""
Class for a generic trainer used for training all the different generative models
"""
import torch
import torch.nn as nn
from Utils.utils import *
from collections import deque, defaultdict
from models import vae
from models.attention import *
import time
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import os
from skimage import io, transform
import scipy.misc as m


class StatesDataset(Dataset):
    """

    Dataset consisting of the frames of the Atari Game-
    Montezuma Revenge

    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.list_files()

    def __len__(self):
        return len(self.images)

    def list_files(self):
        for m in os.listdir(self.root_dir):
            if m.endswith('.jpg'):
                self.images.append(m)

    def __getitem__(self, idx):
        m = self.images[idx]
        image = io.imread(os.path.join( self.root_dir, m))
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Transformations
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.FloatTensor(torch.from_numpy(image).float())}


class Trainer(object):

    """
    Training class for training a beta vae
    with the option of training it with a denoising autoencoder
    similar to how it was trained in DARLA
    """

    def __init__(self, beta, num_d_epochs,
                 generative_model, learning_rate, num_epochs,
                 input_images_folder, batch_size, image_size,
                 random_seed, output_folder, multi_gpu_training=False,
                 use_cuda=True, save_model=True, verbose=True,
                 plot_stats=True, shuffle=True, model_path=None,
                 denoising_autoencoder=None, d_output_folder=None,
                 dae_weights=None):

        """

        :param generative_model: The generative model (eg VAE) to train
        :param learning_rate: learning rate for the optimizer
        :param num_epochs: total number of training steps
        :param random_seed: set the random seed for reproduction of the results
        :param output_folder: output folder for the saved model
        :param use_cuda: use cuda in case of availability of gpu
        :param save_model: save the generative model weights
        :param verbose: print the training statements
        :param plot_stats: plot the stats of training
        :param beta: This hyperparameter decides the disentanglement factor of the vae
        :param model_path: The path to the saved weights of the model
        """

        self.model = generative_model
        self.lr = learning_rate
        self.seed = random_seed
        self.input_images = input_images_folder
        self.num_epochs = num_epochs
        self.output_folder = output_folder
        self.cuda  = use_cuda
        self.multi_gpu = multi_gpu_training
        self.save_model_bool = save_model
        self.verbose = verbose
        self.plot_stats= plot_stats
        self.batch = batch_size
        self.shuffle = shuffle
        self.dataset = StatesDataset(root_dir=self.input_images, transform=
        transforms.Compose([Rescale(image_size), ToTensor()]))
        self.optimizer = optim.Adam(lr=learning_rate, params=self.model.parameters())
        self.beta = beta
        self.model_weights = model_path
        self.latents = []
        self.d_autoencoder = denoising_autoencoder
        self.d_output_folder = d_output_folder
        self.d_optim = optim.Adam(lr=learning_rate, params=self.d_autoencoder.parameters())
        self.dae_weights = dae_weights
        self.num_d_epochs = num_d_epochs


    def get_dataloader(self):
        # Generates the dataloader for the images for training

        dataset_loader = DataLoader(self.dataset,
                                    batch_size=self.batch,
                                    shuffle=self.shuffle)

        return dataset_loader

    # Definition of the loss function -> Defining beta which is used in beta-vae
    def loss_function(self, recon_x, x, mu, logvar, beta, BATCH_SIZE):
        # This is the log p(x|z) defined as the mean squared loss between the
        # reconstruction and the original image
        x.detach_()
        MSE = nn.MSELoss()(recon_x, x)


        # KLD - Kullback liebler divergence -- how much one learned distribution
        # deviate from one another, in this case the learned distribution
        # from the unit Gaussian.

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        mu_sum_sq = (mu * mu).sum(dim=1)
        sigma = logvar.mul(0.5).exp_()
        sig_sum_sq = (sigma * sigma).sum(dim=1)
        log_term = (1 + torch.log(sigma ** 2)).sum(dim=1)
        kldiv = -0.5 * (log_term - mu_sum_sq - sig_sum_sq)
        # Normalize by the same number of elements in reconstruction
        #KLD = KLD / BATCH_SIZE

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian

        # To learn disentangled representations, we use the beta parameter
        # as in the beta-vae
        loss = MSE + beta * kldiv.mean()

        return loss

    def mse_loss(self, input, target):
        return torch.sum((input - target) ^ 2) / input.data.nelement()

    def train_dae(self):

        if self.d_autoencoder is not None:

            for epoch in range(self.num_d_epochs):
                cummulative_loss = 0
                for i_batch, sampled_batch in enumerate(self.get_dataloader()):
                    image = sampled_batch['image']
                    image = Variable(image)
                    self.d_optim.zero_grad()
                    decoded_image, encoded = self.d_autoencoder(image)
                    loss = nn.MSELoss()(decoded_image, image)
                    loss.backward()
                    cummulative_loss += loss.data[0]

                    self.d_optim.step()

                print(cummulative_loss)

            self.save_model(output=self.d_output_folder, model=self.d_autoencoder)

    def train_bvae(self):
        dae_state_dict = torch.load(self.dae_weights)
        dae = vae.DAE(conv_layers=16, conv_kernel_size=3, pool_kernel_size=2,
                                    height=96, width=96, input_channels=3, hidden_dim=64, noise_scale=0)
        dae.load_state_dict(dae_state_dict)
        # Freeze the weights of the denoising autoencoder
        dae.eval()

        for epoch in range(self.num_epochs):
            cummulative_loss = 0
            for i_batch, sampled_batch in enumerate(self.get_dataloader()):
                image = sampled_batch['image']
                image = Variable(image)
                self.optimizer.zero_grad()
                decoded_image, mu, logvar, z = self.model(image)
                decoded_image_encoding = dae.encode(decoded_image)
                image_encoding = dae.encode(image)
                loss = self.loss_function(decoded_image_encoding, image_encoding, mu, logvar,
                                              self.beta, self.batch)
                loss.backward()
                cummulative_loss += loss.data[0]

                self.optimizer.step()

            print(cummulative_loss)

        self.save_model(output=self.output_folder, model=self.model)

    def seed(self, s):
        # Seed everything to make things reproducible
        random.seed = s
        np.random.seed(seed=s)

    def save_model(self, output, model):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the generative model")
        torch.save(
            model.state_dict(),
            '{}/generative_model.pt'.format(output)
        )

    def load_model(self):
        # Load the model from the saved weights file
        if self.model_weights is not None:
            model_state_dict = torch.load(self.model_weights)
            self.model.load_state_dict(model_state_dict)
        else:
            print("Train a model for loading later")

    def inference(self):
        # Do inference on the images
        self.load_model()
        self.model.eval()

        for j, batch in enumerate(self.get_dataloader()):
            images = batch['image']
            index = random.randint(0, self.batch-1)
            i = images[index]
            i = Variable(torch.unsqueeze(i, 0))
            decoded_image, mu, logvar, z = self.model(i)
            self.latents.append(z)
            decoded_image = decoded_image.data.numpy()
            decoded_image = np.squeeze(decoded_image, 0)
            decoded_image = np.transpose(decoded_image, (1, 2, 0))
            path = os.path.join('', str(j) + 'decoded.jpg')
            m.imsave(path, decoded_image)

    def interpolate(self):
        z = self.latents[0]
        self.interpolated_latents = []
        add = np.zeros(shape=z.shape)
        for i in range(-3, 3):
            add[0] = i
            z_new = z + add
            self.interpolated_latents.append(z_new)


if __name__ == '__main__':
    image_size = 96
    seed = 100
    generative_model = vae.VAE(conv_layers=16, z_dimension=32,
                               pool_kernel_size=2, conv_kernel_size=3,
                               input_channels=3, height=96, width=96, hidden_dim=64)
    denoising_autoencoder = vae.DAE(conv_layers=16, conv_kernel_size=3, pool_kernel_size=2,
                                    height=96, width=96, input_channels=3, hidden_dim=64, noise_scale=0.3)
    trainer = Trainer(beta=1, generative_model=generative_model, learning_rate=1e-2,
                      num_epochs=30, input_images_folder='montezuma_resources',
                      image_size=image_size, batch_size=16, output_folder='vae_output/',
                      random_seed=seed, model_path='vae_output/generative_model.pt',
                      denoising_autoencoder=denoising_autoencoder,
                      d_output_folder='denoising_output/', dae_weights='denoising_output/generative_model.pt',
                      num_d_epochs=60)
    trainer.train_dae()
    trainer.train_bvae()

    #Inference
    trainer.inference()