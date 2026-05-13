import os
import torch

# Paths
WORKSPACE = os.getcwd()
DATA_PATH = os.path.join(WORKSPACE, "Data")

AD_PATH = os.path.join(DATA_PATH, "AD_Data")
ASD_PATH = os.path.join(DATA_PATH, "ASD_Data")
ICH_PATH = os.path.join(DATA_PATH, "ICH_Data")

# Dataset specific paths
DATASET_PATH = AD_PATH
IMAGES_PATH = os.path.join(DATASET_PATH, 'Images')
TRAIN_DATASET_PATH = os.path.join(IMAGES_PATH, 'Train')
TEST_DATASET_PATH = os.path.join(IMAGES_PATH, 'Test')

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 15
IMAGE_SIZE = 96
WIN_SIZE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
LEARNING_RATE = 0.01
TEST_RATIO = 0.2
RAND_SEED_DATA = 5

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
