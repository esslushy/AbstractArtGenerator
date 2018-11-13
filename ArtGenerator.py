#imports
from utils import Logger

import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

from torchvision import transforms, datasets
#get data
def cifarData():
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    return datasets.ImageFolder("Seagate Drive", transform=compose)

data = cifarData()
batchSize = 100
dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=True)
numBatches = len(dataLoader)

#make networks
#discrim using conv in the opposite direction 
class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(#adds it in sequential order
            nn.Conv2d(#go from output of deconv back to input  
                in_channels=3, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=2048, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=2048, out_channels=4096, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=4096, out_channels=8192, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(8192),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=8192, out_channels=16384, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(16384),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=16384, out_channels=32768, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32768),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=32768, out_channels=65536, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(65536),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(65536*0.5*0.5, 1),#what the generator takes
            nn.Sigmoid(),#narrows it between 0 (fake) and 1 (real)
        )

    def forward(self, x):#run the network
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 65536*0.5*0.5)
        x = self.out(x)
        return x

class GenerativeNet(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        self.linear = torch.nn.Linear(100, 65536*0.5*0.5)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=65536, out_channels=32768, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32768),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32768, out_channels=16384, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(16384),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16384, out_channels=8192, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(8192),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8192, out_channels=4096, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4096, out_channels=2048, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2048, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,#what it ends in
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()#the paper said to use this

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 65536, 0.5, 0.5)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # Apply Tanh
        return self.out(x)
    
# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n
def initWeights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

# Create Network instances and init weights
generator = GenerativeNet()
generator.apply(initWeights)

discriminator = DiscriminativeNet()
discriminator.apply(initWeights)

# Enable cuda if available. Thanks for the cuda tip cooper
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

# Optimizers
discrimOptimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))#paper said this was best
generatorOptimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss = nn.BCELoss()

# Number of epochs
epochs = 200

def realDataTarget(size):
    '''
    Tensor containing ones, with shape = size
    Use 1 to show real 0 for fake
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fakeDataTarget(size):
    '''
    Tensor containing zeros, with shape = size
    Use 1 to show real 0 for fake
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def trainDiscriminator(optimizer, realData, fakeData):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    predictionReal = discriminator(realData)
    # Calculate error and backpropagate
    errorReal = loss(predictionReal, realDataTarget(realData.size(0)))
    errorReal.backward()

    # 1.2 Train on Fake Data
    predictionFake = discriminator(fakeData)
    # Calculate error and backpropagate
    errorFake = loss(predictionFake, fakeDataTarget(realData.size(0)))
    errorFake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return errorReal + errorFake, predictionReal, predictionFake

def trainGenerator(optimizer, fakeData):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fakeData)
    # Calculate error and backpropagate
    error = loss(prediction, realDataTarget(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

testSamples = 16
testNoise = noise(testSamples)

logger = Logger(model_name='DCGAN', data_name='CIFAR10')

for epoch in range(epochs):
    for nBatch, (realBatch,_) in enumerate(dataLoader):
        
        # 1. Train Discriminator
        realData = Variable(realBatch)
        if torch.cuda.is_available(): realData = realData.cuda()
        # Generate fake data
        fakeData = generator(noise(realData.size(0))).detach()
        # Train D
        discrimError, discrimPredictionReal, discrimPredictionFake = train_discriminator(discrimOptimizer, 
                                                                realData, fakeData)

        # 2. Train Generator
        # Generate fake data
        fakeData = generator(noise(realBatch.size(0)))
        # Train G
        generatorError = trainGenerator(generatorOptimizer, fakeData)
        # Log error
        logger.log(discrimError, generatorError, epoch, nBatch, numBatches)
        
        # Display Progress
        if (nBatch) % 100 == 0:
            display.clear_output(True)
            # Display Images
            test_images = generator(test_noise).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, nBatch, numBatches)
            # Display status Logs
            logger.display_status(
                epoch, epochs, nBatch, numBatches,
                discrimError, generatorError, discrimPredictionReal, discrimPredictionFake
            )
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)