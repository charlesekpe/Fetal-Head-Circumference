import torch
from torchvision import transforms

DIR = 'training_set/'
NUM_WORKERS = 6
RANDOM_STATE = 0
validation_set_size = 0.3
LEARNING_RATE = 0.0077
EPOCHS = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Loads to the GPU 
x_size = (572, 572)
y_size = (572, 572)

tx_X = transforms.Compose([ transforms.Resize(x_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                              ])
tx_Y = transforms.Compose([ transforms.Resize(y_size),
                              transforms.ToTensor()
                              ])
transform_xy = {'x': tx_X,
                'y': tx_Y}
