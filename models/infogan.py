import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

USE_CUDA = torch.cuda.is_available()


class InfoGAN(object):
    # The InfoGAN class consisting of the Generator, Discriminator and the Recognizer
    def __init__(self, generator, discriminator,
                 dataset, num_epochs,
                 random_seed, shuffle, use_cuda,
                 tensorboard_summary_writer,
                 output_folder,
                 generator_lr, discriminator_lr, batch_size):

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.seed = random_seed
        self.batch = batch_size
        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.generator = generator
        self.discriminator = discriminator
        self.tb_writer = tensorboard_summary_writer
        self.output_folder = output_folder

        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()


        self.gen_optim = Adam(self.generator.parameters(), lr=generator_lr)
        self.dis_optim = Adam(self.discriminator.parameters(), lr=discriminator_lr)


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

    # Loss Function
    def loss(self):

        # Discriminator loss
        criterionD = nn.BCELoss()
        criterionQ_categorical = nn.CrossEntropyLoss()
        criterionQ_continuos  = nn.MSELoss()

        return criterionD, criterionQ_categorical, criterionQ_continuos

    # Noise Sample Generator
    def _noise_sample(self, cat_c, con_c, noise, bs):
        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs), idx] = 1.0

        cat_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, cat_c, con_c], 1).view(-1, 74)

        return z, idx

    def linear_annealing_variance(self, std, epoch):
        # Reduce the standard deviation over the epochs
        if std > 0:
            std -= epoch*0.1
        else:
            std = 0
        return std

    def train(self):
        real_x = torch.FloatTensor(self.batch_size, 3, 128, 128)
        labels = torch.FloatTensor(self.batch_size)
        cat_c = torch.FloatTensor(self.batch_size, 10)
        con_c = torch.FloatTensor(self.batch_size, 2)
        noise = torch.FloatTensor(self.batch_size, 62)

        cat_c = Variable(cat_c)
        con_c = Variable(con_c)
        noise = Variable(noise)


        labels = Variable(labels)
        labels.requires_grad = False

        criterionD, criterion_cat, criterion_cont = self.loss()

        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        print(c1.shape)

        idx = np.arange(10).repeat(self.batch_size)
        one_hot = np.zeros((10))
        one_hot[1] = 1
        fix_noise = torch.Tensor(62).uniform_(-1, 1)

        for epoch in range(self.num_epochs):
            std = 1
            for num_iters, batch_data in enumerate(self.get_dataloader()):

                # Real Part
                self.dis_optim.zero_grad()

                x = batch_data['image']
                bs = x.size(0)

                x = Variable(x)

                if self.use_cuda:
                    x = x.cuda()
                    real_x = real_x.cuda()
                    labels = labels.cuda()
                    cat_c = cat_c.cuda()
                    con_c = con_c.cuda()
                    noise = noise.cuda()

                real_x.data.resize_(x.size())
                labels.data.resize(bs)
                cat_c.data.resize_(bs, 10)
                con_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 62)

                real_x.data.copy_(x)
                # Add noise to the inputs of the discriminator
                noise = torch.zeros(x.shape)
    #            print(noise.shape)
                noise = torch.normal(means=noise, std=std)
                if self.use_cuda:
                    noise = noise.cuda()

                x += noise
                d_output, recog_cat, recog_cont = self.discriminator(x)
                labels.data.fill_(1)
                loss_real = criterionD(d_output, labels)
                loss_real.backward()

                # Fake Part
                z, idx = self._noise_sample(cat_c, con_c, noise, bs)
                fake_x = self.generator(z)
                fake_x = fake_x + noise
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

                class_ = torch.LongTensor(idx)
                target = Variable(class_)

                if self.use_cuda:
                    target = target.cuda()

                cont_loss = criterion_cont(recog_cont, con_c)*0.1
                cat_loss = criterion_cat(recog_cat, target)*1 # Refer to the paper for the values of lambda

                G_loss = reconstruct_loss + cont_loss + cat_loss
                G_loss.backward()

                self.gen_optim.step()

                std = self.linear_annealing_variance(std=std, epoch=epoch)

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )

                    #noise.data.resize_(100, 62)
                    #cat_c.data.resize_(100, 10)
                    #con_c.data.resize(100, 2)


                    noise.data.copy_(fix_noise)
                    cat_c.data.copy_(torch.Tensor(one_hot))

                    con_c.data.uniform_(-1.0, 1.0)
                    z = torch.cat([noise, cat_c, con_c], 1).view(-1, 74)
                    x_save = self.generator(z)
                    save_image(x_save.data.cpu(), 'infogan/inference/c1.png', nrow=10)

                    #con_c.data.copy_(torch.from_numpy(c2))
                    con_c.data.uniform_(-1.0, 1.0)
                    z = torch.cat([noise, cat_c, con_c], 1).view(-1, 74)
                    x_save = self.generator(z)
                    save_image(x_save.data.cpu(), 'infogan/inference/c2.png', nrow=10)

            self.save_model(output=self.output_folder)

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
            '{}/generator.pt'.format(output)
        )
        torch.save(
            self.discriminator.state_dict(),
            '{}/discriminator.pt'.format(output)
        )


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


class Discriminator_recognizer(nn.Module):

    """
    The discriminator and the recognizer network for the infogan

    This network distinguishes the fake images from the real

    """

    def __init__(self, input_channels, conv_layers,
                 pool_kernel_size, conv_kernel_size,
                 height, width, hidden, cat_dim, cont_dim):

        super(Discriminator_recognizer, self).__init__()

        self.in_channels = input_channels
        self.conv_layers = conv_layers
        self.pool = pool_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.height = height
        self.width = width
        self.hidden = hidden
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim

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

        self.recognizer_output_cont = nn.Linear(in_features=self.hidden, out_features=self.cont_dim)
        self.recognizer_output_cat = nn.Linear(in_features=self.hidden, out_features=self.cat_dim)

        self.softmax_output = nn.Softmax()

        # Weight initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.recognizer_output_cat.weight)
        nn.init.xavier_uniform_(self.recognizer_output_cont.weight)

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

        hidden = self.hidden_layer1(pool2)
        hidden = self.leaky_relu(hidden)

        #feature_mean = hidden

        output = self.output(hidden)
        output = self.sigmoid_output(output)

        cat_output = self.recognizer_output_cat(hidden)
        cat_output = self.softmax_output(cat_output)

        cont_output = self.recognizer_output_cont(hidden)

        return output, cat_output, cont_output


