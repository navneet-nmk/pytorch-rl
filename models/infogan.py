import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

USE_CUDA = torch.cuda.is_available()


class InfoGAN(object):
    # The InfoGAN class consisting of the Generator, Discriminator and the Recognizer
    def __init__(self,
                 conv_layers, conv_kernel_size,
                 generator_input_channels, generator_output_channels,
                 height, width, discriminator_input_channels,
                 discriminator_output_dim, num_epochs,
                 output_dim, categorical_dim, continuous_dim,
                 hidden_dim, pool_kernel_size, generator_lr, discriminator_lr, batch_size):

        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.cat_dim = categorical_dim
        self.cont_dim = continuous_dim
        self.hidden = hidden_dim
        self.height = height
        self.width = width
        self.d_output_dim = discriminator_output_dim
        self.g_input_channels = generator_input_channels
        self.g_output_channels = generator_output_channels
        self.d_input_channels = discriminator_input_channels
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs



        # Generator
        self.generator = Generator(conv_layers=self.conv_layers, conv_kernel_size=self.conv_kernel_size,
                                   input_channels=self.g_input_channels, output_channels=self.g_output_channels)

        # Discriminator
        self.discriminator = Discriminator_recognizer(conv_layers=self.conv_layers, conv_kernel_size=conv_kernel_size,
                                                      input_dim=self.d_input_channels, height=self.height,
                                                      width=self.width, output_dim=self.output_dim,
                                                      categorical_dim=self.cat_dim, continuous_dim=self.cont_dim,
                                                      discriminator_output_dim=self.d_output_dim, hidden_dim=self.hidden,
                                                      pool_kernel_size=self.pool_kernel_size
                                                      )

        self.gen_optim = Adam(self.generator.parameters(), lr=generator_lr)
        self.dis_optim = Adam(self.discriminator.parameters(), lr=discriminator_lr)

    # Loss Function
    def loss(self):

        # Discriminator loss
        criterionD = nn.BCELoss()
        criterionQ_categorical = nn.CrossEntropyLoss()
        criterionQ_continuos  =nn.MSELoss()

        return criterionD, criterionQ_categorical, criterionQ_continuos

    # Noise Sample Generator
    def _noise_sample(self, cat_c, con_c, noise, bs):
        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs), idx] = 1.0

        cat_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, cat_c, con_c], 1).view(-1, 74, 1, 1)

        return z, idx

    def train(self, real_x, dataloader):
        labels = torch.FloatTensor(self.batch_size)
        cat_c = torch.FloatTensor(self.batch_size, 10)
        con_c = torch.FloatTensor(self.batch_size, 2)
        noise = torch.FloatTensor(self.batch_size, 62)

        cat_c = Variable(cat_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        with torch.no_grads():
            labels = Variable(labels)

        criterionD, criterion_cat, criterion_cont = self.loss()

        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

        for epoch in range(self.num_epochs):
            for num_iters, batch_data in enumerate(dataloader, 0):

                # Real Part
                self.dis_optim.zero_grad()

                x, _ = batch_data
                bs = x.size(0)

                real_x.data.resize_(x.size())
                labels.data.resize(bs)
                cat_c.data.resize_(bs, 10)
                con_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 62)

                real_x.data.copy_(x)
                d_output, recog_cat, recog_cont = self.discriminator(x)
                labels.data.fill_(1)
                loss_real = criterionD(d_output, labels)
                loss_real.backward()


                # Fake Part
                z, idx = self._noise_sample(cat_c, con_c, noise, bs)
                fake_x = self.generator(z)
                d_output, recog_cat, recog_cont = self.discriminator(fake_x.detach())
                labels.data.fill_(0)
                loss_fake = criterionD(d_output, labels)
                loss_fake.backward()

                D_loss = loss_real+loss_fake
                self.dis_optim.step()

                # Generator and Recognizer Part
                d_output, recog_cat, recog_cont = self.discriminator(fake_x)
                labels.data.fill_(1.0)
                reconstruct_loss = criterionD(d_output, labels)

                cont_loss = criterion_cont(recog_cont, con_c)*0.1
                cat_loss = criterion_cat(recog_cat, cat_c)*1 # Refer to the paper for the values of lambda

                G_loss = reconstruct_loss + cont_loss + cat_loss
                G_loss.backward()

                self.gen_optim.step()

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )

    def to_cuda(self):
        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()

    def save_model(self, output):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the generator and discriminator")
        torch.save(
            self.generator.state_dict(),
            '{}/generator.pkl'.format(output)
        )
        torch.save(
            self.discriminator.state_dict(),
            '{}/discriminator.pkl'.format(output)
        )

class Generator(nn.Module):

    def __init__(self, conv_layers,
                 conv_kernel_size, input_channels, output_channels):
        super(Generator, self).__init__()


        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Generator input -> Noise Vector+Latent Codes

        self.conv1 = nn.ConvTranspose2d(in_channels=self.input_channels,
                               out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv4 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv6 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.output_channels, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size-2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape : Noise dimension + latent code dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)

        x = self.output(x)
        return x


class Discriminator_recognizer(nn.Module):

    # The discriminator and recognizer network for the infogan

    def __init__(self, conv_layers, conv_kernel_size, height, width, input_dim,
                 output_dim, categorical_dim, continuous_dim, pool_kernel_size,
                 hidden_dim, discriminator_output_dim):
        super(Discriminator_recognizer, self).__init__()

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.height = height
        self.width = width
        self.input_channels = input_dim
        self.cat_dim = categorical_dim
        self.cont_dim = continuous_dim
        self.output_dim = output_dim
        self.pool = pool_kernel_size
        self.hidden = hidden_dim
        self.d_output_dim = discriminator_output_dim

        # Shared Network
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=self.conv_layers,
                               padding=0, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               padding=0, kernel_size=self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               padding=0, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               padding=0, kernel_size=self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv5 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*4,
                               padding=0, kernel_size=self.kernel_size)
        self.conv6 = nn.Conv2d(in_channels=self.conv_layers*4, out_channels=self.conv_layers*4,
                               padding=0, kernel_size=self.kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool)

        height = self.height//8
        width = self.width//8

        self.linear1 = nn.Linear(in_features=height*width*self.conv_layers*4, out_features=self.hidden)
        self.discriminator_output = nn.Linear(in_features=self.hidden, out_features=self.d_output_dim)
        self.recognizer_output_cont = nn.Linear(in_features=self.hidden, out_features=self.cont_dim)
        self.recognizer_output_cat = nn.Linear(in_features=self.hidden, out_features=self.cat_dim)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, h, w, c = x.shape

        # Input to this network is the output of the generator
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view((b, -1))

        x = self.linear1(x)

        discriminator_output = self.discriminator_output(x)
        recognizer_output_cont = self.recognizer_output_cont(x)
        rcat = self.recognizer_output_cat(x)
        recognizer_output_cat = self.softmax(rcat)

        return discriminator_output, recognizer_output_cont, recognizer_output_cat










