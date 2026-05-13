import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from utils import check_and_create_dir
from dataset import prepare_dataframes
from train import train_model
from evaluate import evaluate_and_interpret

def main():
    # 1. Initialize Directories
    for path in [DATA_PATH, AD_PATH, ASD_PATH, ICH_PATH]:
        check_and_create_dir(path)
        
    # 2. Prepare Data
    df_train, df_valid, df_test, oh_train, oh_valid, oh_test, dict_label2num, dict_num2label = prepare_dataframes()
    
    # 3. Model, Loss, and Optimizer Setup
    # (Initialize your specific MONAI architecture, dataloaders, and loss function here)
    # model = ... 
    # model.to(DEVICE)
    # loss_function = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # train_loader, val_loader, test_loader = ...
    
    # 4. Train Model
    # train_loss, val_acc = train_model(model, train_loader, val_loader, loss_function, optimizer)
    
    # 5. Evaluate and Interpret
    # Load best model
    # model.load_state_dict(torch.load("best_metric_model_classification3d.pth"))
    # evaluate_and_interpret(model, test_loader, test_ds, dict_label2num, dict_num2label)

if __name__ == "__main__":
    main()
