import random
import collections
import os
import wandb
from dotenv import load_dotenv, find_dotenv
import json
import tensorflow as tf
from tqdm import tqdm
import numpy as np
load_dotenv(find_dotenv())

WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ENTITY = os.environ["WANDB_ENTITY"]


def load_raw_wandb_data():
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="grab-raw-data")
    annotation_art = run.use_artifact("annotations:latest")
    annotation_file = os.path.join(
        annotation_art.download(), "captions_train2014.json")

    images_art = run.use_artifact("images:latest")
    images_path = images_art.download()

    run.finish()
    return annotation_file, images_path


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def process_data():
    annotation_file, images_path = load_raw_wandb_data()

    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="log-inception-features")

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
    train_image_paths = image_paths[:6000]
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

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
        load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32)

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

    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="log-input-dataset")
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    # We will override the default standardization of TextVectorization to preserve
    # "<>" characters, so we preserve the tokens for the <start> and <end>.
    def standardize(inputs):
        inputs = tf.strings.lower(inputs)
        return tf.strings.regex_replace(inputs,
                                        r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

    # Max word count for a caption.
    max_length = 50
    # Use the top 5000 words for a vocabulary.
    vocabulary_size = 5000
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length)
    # Learn the vocabulary from the caption data.
    tokenizer.adapt(caption_dataset)

    # Create the tokenized vectors
    cap_vector = caption_dataset.map(lambda x: tokenizer(x))

    # Create mappings for words to indices and indicies to words.
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:
                                                      slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    # Feel free to change these parameters according to your system's configuration

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    # Load the numpy files
    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int64]),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(dataset)

    # TODO: Save this tf.Dataset into a tfrecord for model training
    run.finish()

    return None


if __name__ == "__main__":
    process_data()
