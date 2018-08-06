"""
This script contains an implementation of the CVAEGAN paper.

I have not used class conditional gan and vae for this implementation.
I have just used the the ideas from the paper for stable training of the GAN
and VAE parts of the network.


For the GAN stability tricks used please refer to
https://github.com/soumith/ganhacks

"""

import torch
print(torch.__version__)
import torch.nn as nn
import os
import scipy.misc as m
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()


class Encoder(nn.Module):

    """
    The encoder network in the cvae-gan pipeline

    Given an image, this network returns the latent encoding
    for the image.

    """

    def __init__(self, conv_layers, conv_kernel_size,
                 latent_space_dim, hidden_dim, use_cuda,
                 height, width, input_channels,pool_kernel_size):
        super(Encoder, self).__init__()

        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.z_dim = latent_space_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.height = height
        self.width = width
        self.in_channels = input_channels
        self.pool_size = pool_kernel_size

        # Encoder Architecture

        # 1st Stage
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv_layers*2)
        # Use strided convolution instead of maxpooling for generative models.
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # 2nd Stage
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*4,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(self.conv_layers*4)
        # Use strided convolution instead of maxpooling for generative models.
        self.pool2  = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # Linear Layer
        self.linear1 = nn.Linear(in_features=self.height//16*self.width//16*self.conv_layers*4, out_features=self.hidden_dim)
        self.latent_mu = nn.Linear(in_features=self.hidden_dim, out_features=self.z_dim)
        self.latent_logvar = nn.Linear(in_features=self.hidden_dim, out_features=self.z_dim)
        self.relu = nn.ReLU(inplace=True)

        # The stability of the GAN Game suffers from the problem of sparse gradients
        # Therefore, try to use LeakyRelu instead of relu
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.latent_mu.weight)
        nn.init.xavier_uniform_(self.latent_logvar.weight)


    def encode(self, x):
        # Encoding the input image to the mean and var of the latent distribution
        bs, _, _, _ = x.shape

        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.relu(conv2)
        #pool = self.pool(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.relu(conv4)
        #pool2 = self.pool2(conv4)

        pool2 = conv4.view((bs, -1))

        linear = self.linear1(pool2)
        linear = self.relu(linear)
        mu = self.latent_mu(linear)
        logvar = self.latent_logvar(linear)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):

    """
    The generator/decoder in the CVAE-GAN pipeline

    Given a latent encoding or a noise vector, this network outputs an image.

    """

    def __init__(self, latent_space_dimension, conv_kernel_size,
                 conv_layers, hidden_dim, height, width, input_channels):
        super(Generator, self).__init__()

        self.z_dimension = latent_space_dimension
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.hidden = hidden_dim
        self.height = height
        self.width = width
        self.input_channels = input_channels

        # Decoder/Generator Architecture
        self.linear_decoder = nn.Linear(in_features=self.z_dimension,
                                        out_features=self.height//16 * self.width//16 * self.conv_layers*4)
        #self.bnl = nn.BatchNorm2d(se)

        # Deconvolution layers
        self.conv1 = nn.ConvTranspose2d(in_channels=self.conv_layers*4,
                                        out_channels=self.conv_layers*4, kernel_size=self.conv_kernel_size,
                                        stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv_layers*4)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.conv_layers*4, out_channels=self.conv_layers*2,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv_layers*2)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)

        self.conv4 = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.bn4 = nn.BatchNorm2d(self.conv_layers)

        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.input_channels,
                                kernel_size=self.conv_kernel_size-1, stride=1)

        self.relu = nn.ReLU(inplace=True)

        # The stability of the GAN Game suffers from the problem of sparse gradients
        # Therefore, try to use LeakyRelu instead of relu
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # Use dropouts in the generator to stabilize the training
        self.dropout = nn.Dropout()

        self.sigmoid_output = nn.Sigmoid()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.linear_decoder.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, z):
        z  = self.linear_decoder(z)
        z = self.leaky_relu(z)

        z =  z.view((-1, self.conv_layers*4, self.height//16, self.width//16))

        z = self.conv1(z)
        z = self.bn1(z)
        z = self.leaky_relu(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.leaky_relu(z)
        #z = self.dropout(z)

        z = self.conv3(z)
        z = self.bn3(z)
        z = self.leaky_relu(z)
        z = self.conv4(z)
        z = self.bn4(z)
        z = self.leaky_relu(z)
        #z = self.dropout(z)

        output = self.output(z)
        output = self.sigmoid_output(output)

        return output


class Discriminator(nn.Module):

    """
    The discriminator network in the CVAEGAN pipeline

    This network distinguishes the fake images from the real

    """

    def __init__(self, input_channels, conv_layers,
                 pool_kernel_size, conv_kernel_size,
                 height, width, hidden):

        super(Discriminator, self).__init__()

        self.in_channels = input_channels
        self.conv_layers = conv_layers
        self.pool = pool_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.height = height
        self.width = width
        self.hidden = hidden

        # Discriminator architecture
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv_layers*2)
        # Use strided convolution in place of max pooling
        self.pool_1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv3 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*4,
                               kernel_size=self.conv_kernel_size, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(self.conv_layers*4)
        # Use strided convolution in place of max pooling
        self.pool_2 = nn.MaxPool2d(kernel_size=self.pool)

        self.relu = nn.ReLU(inplace=True)

        # The stability of the GAN Game suffers from the problem of sparse gradients
        # Therefore, try to use LeakyRelu instead of relu
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # Fully Connected Layer
        self.hidden_layer1 = nn.Linear(in_features=self.height//16*self.width//16*self.conv_layers*4,
                                       out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=1)
        self.sigmoid_output = nn.Sigmoid()

        # Weight initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.output.weight)

        # Dropout layer
        self.dropout = nn.Dropout()

    def forward(self, input):

        conv1 = self.conv1(input)
        conv1 = self.bn1(conv1)
        conv1 = self.leaky_relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.leaky_relu(conv2)
        #pool1 = self.pool_1(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.leaky_relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.leaky_relu(conv4)
        #pool2 = self.pool_2(conv4)

        pool2 = conv4.view((-1, self.height//16*self.width//16*self.conv_layers*4))

        feature_mean = pool2

        hidden = self.hidden_layer1(pool2)
        hidden = self.leaky_relu(hidden)

        #feature_mean = hidden

        output = self.output(hidden)
        output = self.sigmoid_output(output)

        return output, feature_mean


class CVAEGAN(object):

    """

    The complete CVAEGAN Class containing the following models

    1. Encoder
    2. Generator/Decoder
    3. Discriminator

    """

    def __init__(self, encoder,
                 batch_size,
                 num_epochs,
                 random_seed, dataset,
                 generator, discriminator,
                 encoder_lr, generator_lr,
                 discriminator_lr, use_cuda,
                 output_folder, test_dataset,
                 inference_output_folder,
                 tensorboard_summary_writer,
                 encoder_weights=None, generator_weights=None,
                 shuffle=True,
                 discriminator_weights=None):

        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator

        self.shuffle = shuffle
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.e_lr = encoder_lr
        self.g_lr = generator_lr
        self.d_lr = discriminator_lr
        self.seed = random_seed
        self.batch = batch_size
        self.num_epochs = num_epochs
        self.output_folder = output_folder
        self.inference_output_folder = inference_output_folder

        self.e_optim = optim.Adam(lr=self.e_lr, params=self.encoder.parameters())
        self.g_optim = optim.Adam(lr=self.g_lr, params=self.generator.parameters())
        # GAN stability trick
        self.d_optim = optim.Adam(lr=self.d_lr, params=self.discriminator.parameters())

        self.encoder_weights = encoder_weights
        self.generator_weights = generator_weights
        self.discriminator_weights = discriminator_weights
        self.use_cuda = use_cuda

        if use_cuda:
            # Use cuda for GPU utilization
            self.encoder = self.encoder.cuda()
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        # Tensorboard logger
        self.tb = tensorboard_summary_writer

    def set_seed(self):
        # Set the seed for reproducible results
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def get_dataloader(self):
        # Generates the dataloader for the images for training

        dataset_loader = DataLoader(self.dataset,
                                    batch_size=self.batch,
                                    shuffle=self.shuffle)

        return dataset_loader

    def get_test_dataloader(self):
        # Generates the dataloader for the images for testing
        dataset_loader = DataLoader(self.test_dataset,
                                    batch_size=self.batch)
        return dataset_loader

    def save_model(self, output, model):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the cvaegan model")
        torch.save(
            model.state_dict(),
            '{}/cvaegan.pt'.format(output)
        )

    def klloss(self, mu, logvar):
        # Kullback Liebler divergence loss for the VAE
        mu_sum_sq = (mu*mu).sum(dim=1)
        sigma = logvar.mul(0.5).exp_()
        sig_sum_sq = (sigma * sigma).sum(dim=1)
        log_term = (1 + torch.log(sigma ** 2)).sum(dim=1)
        kldiv = -0.5 * (log_term - mu_sum_sq - sig_sum_sq)

        return kldiv.mean()

    def discriminator_loss(self, x, recon_x, recon_x_noise, std):
        labels_x = torch.FloatTensor(x.shape[0])
        labels_recon_x = torch.FloatTensor(recon_x.shape[0])
        labels_recon_x_noise = torch.FloatTensor(recon_x_noise.shape[0])

        # Labels for the real images are 1 and for the fake are 0
        labels_x.data.fill_(1)
        labels_recon_x.data.fill_(0)
        labels_recon_x_noise.data.fill_(0)

        if self.use_cuda:
            labels_x = labels_x.cuda()
            labels_recon_x = labels_recon_x.cuda()
            labels_recon_x_noise = labels_recon_x_noise.cuda()

        # Adding instance noise to improve the stability of the
        # Discriminator

        mean = torch.zeros(x.shape)
        noise_x = torch.normal(mean=mean, std=std)
        noise_recon = torch.normal(mean=mean, std=std)
        noise_recon_noise = torch.normal(mean=mean, std=std)

        if self.use_cuda:
            noise_x = noise_x.cuda()
            noise_recon = noise_recon.cuda()
            noise_recon_noise = noise_recon_noise.cuda()

        x = x+noise_x
        recon_x = recon_x + noise_x
        recon_x_noise = recon_x_noise + noise_x

        o_x, _ = self.discriminator(x)
        o_x_recon, _ = self.discriminator(recon_x.detach())
        o_x_recon_noise, _ = self.discriminator(recon_x_noise.detach())

        loss_real = nn.BCELoss()(o_x, labels_x)
        loss_fake = nn.BCELoss()(o_x_recon, labels_recon_x)
        loss_fake_noise = nn.BCELoss()(o_x_recon_noise, labels_recon_x_noise)
        loss = torch.mean(loss_fake+loss_fake_noise+loss_real)
        return loss

    def generator_discriminator_loss(self, x,
                                     recon_x_noise, recon_x,
                                     lambda_1, lambda_2, std):

        mean = torch.zeros(x.shape)
        noise_x = torch.normal(mean=mean, std=std)
        noise_recon = torch.normal(mean=mean, std=std)
        noise_recon_noise = torch.normal(mean=mean, std=std)

        if self.use_cuda:
            noise_x = noise_x.cuda()
            noise_recon = noise_recon.cuda()
            noise_recon_noise = noise_recon_noise.cuda()

        x = x + noise_x
        recon_x = recon_x + noise_x
        recon_x_noise = recon_x_noise + noise_x

        # Generator Discriminator loss
        _, fd_x = self.discriminator(x)
        _, fd_x_noise = self.discriminator(recon_x_noise)

        fd_x = torch.mean(fd_x, 0)
        fd_x_noise = torch.mean(fd_x_noise, 0)

        loss_g_d = nn.MSELoss()(fd_x_noise.detach(), fd_x.detach())

        # Generator Loss
        reconstruction_loss = nn.MSELoss()(recon_x, x)
        _, fd_x_r = self.discriminator(x)
        _, fd_x_f = self.discriminator(recon_x)
        feature_matching_reconstruction_loss = nn.MSELoss()(fd_x_f.detach(), fd_x_r.detach())

        loss_g = reconstruction_loss + feature_matching_reconstruction_loss

        loss = lambda_1*loss_g_d +  lambda_2*loss_g

        return loss, loss_g, loss_g_d

    def sample_random_noise(self, z):
        # Sample a random noise vector for the Generator input
        # Sample from a normal distribution with mean 0 and std 1 (similar to P(z) optimized by the VAE)
        noise = torch.randn(z.shape)
        noise = Variable(noise)
        #noise.data.uniform_(-1.0, 1.0)
        if self.use_cuda:
            noise = noise.cuda()
        return noise

    def linear_annealing_variance(self, std, epoch):
        # Reduce the standard deviation over the epochs
        if std > 0:
            std -= epoch*0.1
        else:
            std = 0
        return std

    def train(self, lambda_1, lambda_2):
        std = 1
        for epoch in range(self.num_epochs):
            cummulative_loss_enocder = 0
            cummulative_loss_discriminator = 0
            cummulative_loss_generator = 0
            std = self.linear_annealing_variance(std, epoch)
            for i_batch, sampled_batch in enumerate(self.get_dataloader()):
                images = sampled_batch['image']
                images = Variable(images)
                if self.use_cuda:
                    images = images.cuda()

                latent_vectors, mus, logvars = self.encoder(images)
                loss_kl = self.klloss(mus, logvar=logvars)

                # Reconstruct images from latent vectors - x_f
                recon_images = self.generator(latent_vectors.detach())

                # Reconstruct images from random noise - x_p
                random_noise = self.sample_random_noise(latent_vectors)
                recon_images_noise = self.generator(random_noise)


                if epoch%5 ==0:
                    self.save_image_tensor(reconstructed_images=recon_images, output=self.inference_output_folder,
                                           batch_number=str(epoch)+ '_'+str(i_batch))

                # Discriminator Loss with standard deviation
                loss_d = self.discriminator_loss(x=images, recon_x=recon_images,
                                                 recon_x_noise=recon_images_noise,
                                                 std=std)

                cummulative_loss_discriminator += loss_d


                # Generator Loss
                loss_g, l_g, l_g_d = self.generator_discriminator_loss(x=images, recon_x_noise=recon_images_noise,
                                                           recon_x=recon_images, lambda_1=2, lambda_2=1, std=std)


                cummulative_loss_generator += loss_g

                # Encoder Loss
                loss_e = lambda_1*loss_kl + lambda_2*l_g.detach()

                cummulative_loss_enocder += loss_e

                # Make the gradient updates
                self.d_optim.zero_grad()
                loss_d.backward()
                self.d_optim.step()

                self.g_optim.zero_grad()
                loss_g.backward()
                self.g_optim.step()

                self.e_optim.zero_grad()
                loss_e.backward()
                self.e_optim.step()

            print('Loss Encoder for ', str(epoch), ' is ',
                  cummulative_loss_enocder/575)
            print('Loss Generator for ', str(epoch), ' is ',
                  cummulative_loss_generator/575)
            print('Loss Discriminator for ', str(epoch), ' is ',
                  cummulative_loss_discriminator/575)

            # Log the data onto tensorboard
#            self.tb.write('encoder/loss', cummulative_loss_enocder/len(self.get_dataloader()), epoch)
 #           self.tb.write('generator/loss', cummulative_loss_generator / len(self.get_dataloader()), epoch)
  #          self.tb.write('discriminator/loss', cummulative_loss_discriminator / len(self.get_dataloader()), epoch)

        # Save the models
        self.save_model(output=self.output_folder+'encoder/', model=self.encoder)
        self.save_model(output=self.output_folder+'generator/', model=self.generator)
        self.save_model(output=self.output_folder+'discriminator/', model=self.discriminator)

        # Export and close the Tb Writer
        tb_json = 'tb.json'
        path = os.path.join(self.output_folder, tb_json)
        self.tb.export(path, close_writer=True)

    def load_model(self, weights, model):
        # Load the model from the saved weights file
        if self.use_cuda:
            model_state_dict = torch.load(weights)
        else:
            model_state_dict = torch.load(weights, map_location='cpu')
        model.load_state_dict(model_state_dict)
        return model

    def save_image_tensor(self, reconstructed_images, output, batch_number):
        for i, r_i in enumerate(reconstructed_images):
            decoded_image = r_i.data.cpu().numpy()
            #decoded_image = np.squeeze(decoded_image, 0)
            decoded_image = np.transpose(decoded_image, (1, 2, 0))
            path = os.path.join(output, str(batch_number)+ '_' +str(i) + '.jpg')
            m.imsave(path, decoded_image)

    def inference(self):
        if self.encoder_weights is not None:
            self.encoder = self.load_model(weights=self.encoder_weights, model=self.encoder)
        if self.generator_weights is not None:
            self.generator = self.load_model(weights=self.generator_weights, model=self.generator)
        if self.discriminator is not None:
            self.discriminator = self.load_model(weights=self.discriminator_weights, model=self.discriminator)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        # Set the models in evaluation mode
        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()

        for i_batch, sampled_batch in tqdm(enumerate(self.get_test_dataloader())):
            images = sampled_batch['image']
            images = Variable(images)
            if self.use_cuda:
                images = images.cuda()
            latent_vectors, mus, logvars = self.encoder(images)
            z  = self.sample_random_noise(latent_vectors)
            # Reconstruct images from latent vectors
            reconstructed_images = self.generator(z)
            t, _ = self.discriminator(reconstructed_images)
            t, _ = self.discriminator(images)
            # Save the reconstructed images
            self.save_image_tensor(reconstructed_images=reconstructed_images,
                                   output=self.inference_output_folder, batch_number=i_batch)
            #self.save_image_tensor(reconstructed_images=images, output=self.inference_output_folder, batch_number=i_batch)

        print("Saved the images to ", self.inference_output_folder)
