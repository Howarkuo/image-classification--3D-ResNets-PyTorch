import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config import DATASET_PATH, TEST_RATIO, RAND_SEED_DATA

def prepare_dataframes():
    df_train = pd.read_csv(os.path.join(DATASET_PATH, 'PatientInfo_Train.csv'))
    df_test = pd.read_csv(os.path.join(DATASET_PATH, 'PatientInfo_Test.csv'))

    # Drop MCI label 
    df_train = df_train[df_train['Label'] != 'MCI'].reset_index(drop=True)
    df_test = df_test[df_test['Label'] != 'MCI'].reset_index(drop=True)

    # Encode labels
    labels = sorted(list(df_train['Label'].unique()), reverse=True)
    dict_label2num = {label: num for num, label in enumerate(labels)}
    dict_num2label = {num: label for num, label in enumerate(labels)}

    df_train['Label_y'] = df_train['Label'].map(dict_label2num)
    df_test['Label_y'] = df_test['Label'].map(dict_label2num)

    # Train/Validation Split
    df_train, df_valid = train_test_split(
        df_train,
        stratify=df_train[['Label', 'Sex']],
        test_size=TEST_RATIO,
        random_state=RAND_SEED_DATA,
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    # One-hot encoding
    onehot_train = torch.nn.functional.one_hot(torch.as_tensor(df_train['Label_y'].tolist())).float()
    onehot_valid = torch.nn.functional.one_hot(torch.as_tensor(df_valid['Label_y'].tolist())).float()
    onehot_test = torch.nn.functional.one_hot(torch.as_tensor(df_test['Label_y'].tolist())).float()

    return df_train, df_valid, df_test, onehot_train, onehot_valid, onehot_test, dict_label2num, dict_num2label

# NOTE: You will need to add your MONAI ImageDataset and DataLoader initializations here 
# depending on your specific image transforms.
