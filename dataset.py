import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import config

##BECAUSE THIS IS TREATED AS A SEGMENTATION TASK, WE CAN USE GPU TO SPEED UP THE MASKING PROCESS, WHILE TRAINING
def mask(image):
    if len(image.shape) == 3:
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imgray = image
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        ellipse = cv2.fitEllipse(c)
        image = cv2.ellipse(image, ellipse, (255,255,255), -1)
    return image

class Data(Dataset):

    def __init__(self, data, train_df, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.train_data, self.val_data = train_test_split(train_df, test_size = config.validation_set_size, 
                                                          random_state = config.RANDOM_STATE)

    def __len__(self):
        if(self.data == 'train'):
            return len(self.train_data)
        elif(self.data == 'validate'):
            return len(self.val_data)
        else:
            return ValueError("No data")

    def __getitem__(self, idx):

        if(self.data == 'train'):
            x_train = config.DIR + self.train_data.iloc[idx, 0]
            y_train = x_train.replace('.png','_Annotation.png')
            x = Image.open(x_train)
            y = Image.fromarray(mask(cv2.imread(y_train))).convert('L') # Convert to gray scale


        elif(self.data == 'validate'):
            x_val = dir + self.val_data.iloc[idx, 0]
            y_val = x_val.replace('.png','_Annotation.png')
            x = Image.open(x_val)
            y = Image.fromarray(mask(cv2.imread(y_val))).convert('L') # Convert to gray scale
      
    
        else:
            return ValueError("No data")
        
        if(self.data == 'train'):
            # Random horizontal flipping
            if np.random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

            # Random vertical flipping
            if np.random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

            # Random rotation
            if np.random.random() > 0.8:
                angle = np.random.choice([-30, -90, -60, -45 -15, 0, 15, 30, 45, 60, 90])
                x = TF.rotate(x, int(angle))
                y = TF.rotate(y, int(angle))

        if self.transform:
            x = self.transform['x'](x)
            y = self.transform['y'](y)
    return x, y

class HeadDataset(Dataset):
    '''
    Dataset to get the circumference
    '''
    def __init__(self, train_df, transform=config.tx_X):
        super().__init__()
        self.transform = transform
        self.data = train_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_train, pixel_size, _ = self.data.iloc[idx]
        x = Image.open(config.DIR + x_train)
        if self.transform:
            x = self.transform(x)
        #return image,  pixel_size and filename(for easy identification)
        return x, pixel_size, x_train