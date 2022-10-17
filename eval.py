import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import config
from model import UNet
from utils import (
    circumference,
    train_df,
    get_head_loader
)

train_running_loss_history = np.load(train_running_loss_history)
validation_running_loss_history = np.load(validation_running_loss_history)

plt.plot(train_running_loss_history, label = 'Train Loss')
plt.plot(validation_running_loss_history, label = 'Validation Loss')
plt.legend()
plt.savefig("loss_graph.png")

model = torch.load('hc18.pth.tar', map_location=torch.device(config.device))

"""
We see that the formula(circumference of an ellipse) is not 100 percent accurate
This is because of the drawback of the low pixel size, and the ultrasound image not being a perfect ellipse
"""
for i in range(5):
    name = config.DIR + train_df.iloc[i, 0].replace('.png', '_Annotation.png')
    image = cv2.imread(name)
    p = circumference(image)
    print(f"Calculated: {p*train_df.iloc[i, 1]}, True: {train_df.iloc[i, 2]}")
    
'''
Calculated: 44.67460851409544, True: 44.3
Calculated: 57.306094540209024, True: 56.81
Calculated: 69.08248740172489, True: 68.75
Calculated: 69.49470778558857, True: 69.0
Calculated: 60.139763536180865, True: 59.81
'''
output_table = pd.DataFrame(columns=['filename', 'predicted_circumference', 'pixel_size'])

loader = get_head_loader()
assert loader.batch_size == 1, "Circumference calculator must be loaded one image at a time"

model.eval()

for idx, data in enumerate(loader):
    img, pixel_size, pict_name = data
    pixel_size = pixel_size.cpu().numpy()[0]
    with torch.no_grad():
        img = img.to(config.device)
        preds = model(img)
    im_Array = np.array(transforms.ToPILImage()(preds[0]))
    pred = circumference(im_Array, annotated=False)
    output_table.loc[idx] = [pict_name[0], pred*pixel_size, pixel_size]
    
model.train()
output_table.to_csv('pred.csv', index=False)