import numpy as np
import pandas as pd
import cv2

from torch.utils.data import DataLoader
import config
from dataset import Data, HeadDataset

train_df = pd.read_csv('training_set_pixel_size_and_HC.csv')

# this function will return numpy array from the transformed tensors which were
# obtained from our train_loader. Just to plot them
def im_converterX(tensor):
    image = tensor.cpu().clone().detach().numpy() # make copy of tensor and converting it to numpy as we will need original later
    image = image.transpose(1,2,0) # swapping axes making (1, 28, 28) image to a (28, 28, 1)

    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) 
    # unnormalizing the image # this also outputs (28, 28, 3) which seems important for plt.imshow
    image = image.clip(0, 1) # to make sure final values are in range 0 to 1 as .ToTensor outputed
    return image

def im_converterY(tensor):
    image = tensor.cpu().clone().detach().numpy() # make copy of tensor and converting it to numpy as we will need original later
    image = image.transpose(1,2,0) # swapping axes making (1, 28, 28) image to a (28, 28, 1)
    image = image * np.array((1, 1, 1)) 
    # unnormalizing the image  not needed# this also outputs (28, 28, 3) which seems important for plt.imshow
    image = image.clip(0, 1) # to make sure final values are in range 0 to 1 as .ToTensor outputed
    return image

def dice_index(y_pred, y_actual):
    smooth = 0.000001
    size_of_batch = y_pred.size(0)
    
    p1 = y_pred.view(size_of_batch, -1)
    p2 = y_actual.view(size_of_batch, -1)
    
    intersection = (p1 * p2).sum()
    
    dice =  ((2.0 * intersection )+ smooth) / (p1.sum() + p2.sum() + smooth)
    return dice

def dice_loss(y_predict, y_train): ## to add in bce looss
  
    dice_loss = 1 -(dice_index(y_predict, y_train))

    return dice_loss

def get_model_loaders(df=train_df):
    train = Data('train', df, transform=config.transform_xy)
    val = Data('validate', df, transform=config.transform_xy)

    train_generator = DataLoader(train, batch_size=2, shuffle=True, num_workers=config.NUM_WORKERS)
    val_generator = DataLoader(val, batch_size=2, shuffle=True, num_workers=config.NUM_WORKERS)
    return train_generator, val_generator
    
def get_head_loader(df=train_df):
    head_ds = HeadDataset(df)
    loader = DataLoader(
        head_ds,
        batch_size = 1,
        shuffle=False,
        num_workers=2
    )
    return loader

def calculate_perimeter(a,b):
    #Calculate the perimeter/circumference of an ellipse using semi axes a and b
    perimeter = np.pi * ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) )
    return perimeter

def fit_ellipse(im):    
    ret, thresh = cv2.threshold(im, 127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_sizes = [len(contour) for contour in contours]
    ellipse = cv2.fitEllipse(contours[np.argmax(contour_sizes)])
    return ellipse
  
def circumference(image, annotated=True):
    if annotated:#if annotated image from training set
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ellipse = cv2.fitEllipse(contours[0])
        axes_a, axes_b = ellipse[1]
        p = calculate_perimeter(axes_a/2, axes_b/2)
        return p
    
    else:
        ellipse = fit_ellipse(image)
        axes_a, axes_b = ellipse[1]
        p = calculate_perimeter(axes_a/2, axes_b/2)
        return p
