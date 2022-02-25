import random
import collections
import os
import wandb
from dotenv import load_dotenv, find_dotenv
import json
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from text_utils import generate_text_artifacts
from image_utils import load_image
from config import subset, batch_size, max_length, vocabulary_size, train_split
from utils import save_to_pickle
load_dotenv(find_dotenv())

WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ENTITY = os.environ["WANDB_ENTITY"]

# TODO: Move to load


def download_wandb_coco2014():
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="download-coco2014", job_type="download")
    annotation_art = run.use_artifact("annotations:latest")
    annotation_file = os.path.join(
        annotation_art.download(), "captions_train2014.json")

    images_art = run.use_artifact("images:latest")
    images_path = images_art.download()

    run.finish()
    return annotation_file, images_path


def load_coco2014(annotation_file, images_path, subset=None):
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="load-coco2014", job_type="load_raw")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = os.path.join(images_path, 'COCO_train2014_' +
                                  '%012d.jpg' % (val['image_id']))
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will
    # lead to 30,000 examples.
    if subset:
        train_image_paths = image_paths[:subset]
    else:
        train_image_paths = image_paths
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    img_names = [os.path.basename(path) for path in img_name_vector]
    wandb_imgs = [wandb.Image(path, caption=cap)
                  for cap, path in zip(train_captions, img_name_vector)]

    img_table = wandb.Table(columns=["name", "image", "caption"])
    for name, image, caption in tqdm(zip(img_names, wandb_imgs, train_captions)):
        img_table.add_data(name, image, caption)
    run.log({"image_table": img_table})
    run.finish()

    return train_captions, img_name_vector

# TODO: Injest the logged wandb.Table
# TODO: Remove batching so cna log embeddings into table for projection viewer
# TODO: Read data from logged load data step


def generate_and_log_inception_features(img_name_vector, batch_size=32):
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="log-inception-features", job_type="data_process")

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    inception_v3_extracted_features_dir = os.path.join(".",
                                                       "inception_v3_extracted_features")
    if not os.path.exists(inception_v3_extracted_features_dir):
        os.makedirs(inception_v3_extracted_features_dir)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            path_to_store = os.path.join(
                inception_v3_extracted_features_dir, os.path.basename(path_of_feature))
            np.save(path_to_store, bf.numpy())

    # TODO: Log the trained features into a Table alongside an image for the embedding projector

    inception_features_art = wandb.Artifact(
        name="inception_v3", type="image_features")
    inception_features_art.add_dir(inception_v3_extracted_features_dir)
    run.log_artifact(inception_features_art)
    run.finish()

    return None

# TODO: Read from logged load data step
# TODO: Read directly from wandb.Table
# TODO: Move generate_text_artifacts call here?


def generate_and_log_caption_dataset(train_captions):
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="log-caption-dataset", job_type="data_process")
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)
    caption_dataset_file = os.path.join(".", "caption_dataset.pkl")
    save_to_pickle(train_captions, caption_dataset_file)
    caption_dataset_art = wandb.Artifact(
        name="caption_dataset", type="dataset")
    caption_dataset_art.add_file(caption_dataset_file)
    run.log_artifact(caption_dataset_art)
    run.finish()

    return caption_dataset

# TODO: Read data from wandb.Table?
# TODO: Generate tables for train and test here for visualization purposes


def generate_and_log_test_train_split(img_name_vector, cap_vector):
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="log-test-train-split", job_type="data_process")

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*train_split)
    img_name_train_keys, img_name_val_keys = img_keys[:
                                                      slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for img_path in img_name_train_keys:
        imgt = os.path.basename(img_path)
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for img_path in img_name_val_keys:
        imgv = os.path.basename(img_path)
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    split_art_dir = os.path.join(".", "split_data")
    if not os.path.exists(split_art_dir):
        os.makedirs(split_art_dir)

    save_to_pickle(img_name_train, os.path.join(
        split_art_dir, "img_name_train.pkl"))
    save_to_pickle(cap_train, os.path.join(split_art_dir, "cap_train.pkl"))
    save_to_pickle(img_name_val, os.path.join(
        split_art_dir, "img_name_val.pkl"))
    save_to_pickle(cap_val, os.path.join(
        split_art_dir, "cap_val.pkl"))

    split_art = wandb.Artifact(name="split", type="dataset")
    split_art.add_dir(split_art_dir)
    run.log_artifact(split_art)

    run.finish()

    return img_name_train, cap_train, img_name_val, cap_val

# TODO: Remove all passed variables to make wandb connections in graph view more apparent


def process_data():
    # TODO: Move this into load_coco2014 for better graph view
    annotation_file, images_path = download_wandb_coco2014()
    train_captions, img_name_vector = load_coco2014(
        annotation_file, images_path, subset=subset)
    generate_and_log_inception_features(
        img_name_vector, batch_size=batch_size)
    # TODO: Move generate text artifacts into this function instead. return the cap vector instead of caption dataset
    caption_dataset = generate_and_log_caption_dataset(train_captions)
    _, cap_vector, _, _ = generate_text_artifacts(
        caption_dataset, max_length=max_length, vocabulary_size=vocabulary_size, return_mapping=False)
    generate_and_log_test_train_split(img_name_vector, cap_vector)

    # TODO: Save this tf.Dataset into a tfrecord for model training

    # train
    # -tokenizer
    # -word_to_index
    # -dataset
    # eval
    # -max_length
    # -attention_features_shape
    # -load_image
    # -decoder
    # -encoder
    # -word_to_index
    # -index_to_word
    # -img_name_val
    # -cap_val

    return None


if __name__ == "__main__":
    process_data()
