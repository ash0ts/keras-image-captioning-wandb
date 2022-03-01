import os
# WANDB
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ENTITY = os.environ["WANDB_ENTITY"]

# log_raw_data.py
annotation_folder = '/annotations/'
image_folder = '/train2014/'

# process_data.py
subset = 6000
batch_size = 32
max_length = 50
vocabulary_size = 5000
train_split = 0.8

# train.py
EPOCHS = 20
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
features_shape = 2048
attention_features_shape = 64
