from tqdm import tqdm
import numpy as np
import torch
import config
from utils import (
    get_model_loaders,
    dice_loss
)
from model import UNet

def train_fn(model, criterion, optimizer, train_generator, val_generator):
    train_running_loss = 0.0
    validation_running_loss = 0.0

    model.train()

    for ith_batch, sample_batched in enumerate(tqdm(train_generator)):
        X_train = sample_batched[0].to(config.device)
        y_train = sample_batched[1].to(config.device)
        optimizer.zero_grad()
        y_pred = model(X_train)

        loss = 0.30 * dice_loss(y_pred, y_train) +  0.70 * criterion(y_pred, y_train)
        #loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

    with torch.no_grad():    
        model.eval()

        for ith_batch, sample_batched in enumerate(tqdm(val_generator)):
            X_val = sample_batched[0].to(config.device)
            y_val = sample_batched[1].to(config.device)



            y_out = model(X_val)
            out_val = (y_out + 0.5).int().float()

            val_loss = 0.3 * dice_loss(out_val, y_val)  + 0.7 * criterion(y_out, y_val)

            validation_running_loss += val_loss.item()
    return train_running_loss, validation_running_loss
            
def main():
    model = UNet(1, 1).to(config.device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)
    
    train_generator, val_generator = get_model_loaders()
    train_running_loss_history = []
    validation_running_loss_history =[]

    for e in range(config.EPOCHS):
        train_running_loss, validation_running_loss = train_fn(model, criterion, optimizer, train_generator, val_generator)
        print("================================================================================")
        print("Epoch {} completed".format(e + 1))

        train_epoch_loss = train_running_loss / len(train_generator)
        validation_epoch_loss = validation_running_loss / len(val_generator)

        print("Average train loss is {}: ".format(train_epoch_loss))
        print("Average validation loss is {}".format(validation_epoch_loss))
        print("================================================================================")
        train_running_loss_history.append(train_epoch_loss)
        validation_running_loss_history.append(validation_epoch_loss)
        print("Saving model")
        torch.save(model, 'hc18.pth.tar')
        np.save("train_running_loss_history.npy", train_running_loss_history)
        np.save("validation_running_loss_history.npy", validation_running_loss_history)
    torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
