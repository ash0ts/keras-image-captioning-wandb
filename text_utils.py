import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
import os
import wandb
from utils import save_to_pickle
load_dotenv(find_dotenv())

WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ENTITY = os.environ["WANDB_ENTITY"]


# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.
def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs,
                                    r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


def generate_text_artifacts(text_dataset,
                            max_length=50, vocabulary_size=5000, return_vector=True, return_mapping=True):
    # run = wandb.init(project=WANDB_PROJECT,
    #                  entity=WANDB_ENTITY, name="log-text-artifacts")
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length)
    # Learn the vocabulary from the caption data.
    tokenizer.adapt(text_dataset)

    # Create the tokenized vectors
    # TODO: remove this jank way to make sure i dont rerun the tokenizer over the text data for training
    if return_vector:
        cap_vector = text_dataset.map(lambda x: tokenizer(x))
    else:
        cap_vector = None

    # Create mappings for words to indices and indicies to words.
    # move to train or eval and base it on the tokenizer
    # TODO: remove this jank way to make sure i dont generate these for data processing
    if return_mapping:
        word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary())
        index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True)
    else:
        word_to_index = None
        index_to_word = None

    # text_art_dir = os.path.join(".", "text_artifacts")
    # if not os.path.exists(text_art_dir):
    #     os.makedirs(text_art_dir)

    # CANT LOG THESE MUST CONSTRUCT FROM THIS IN RUNTIME
    # save_to_pickle(tokenizer, os.path.join(text_art_dir, "tokenizer.pkl"))
    # save_to_pickle(cap_vector, os.path.join(cap_vector, "cap_vector.pkl"))
    # save_to_pickle(word_to_index, os.path.join(
    #     word_to_index, "word_to_index.pkl"))
    # save_to_pickle(index_to_word, os.path.join(
    #     index_to_word, "index_to_word.pkl"))

    # text_art = wandb.Artifact(name="text_artifacts", type="text")
    # text_art.add_dir(text_art_dir)
    # run.log_artifact(text_art)

    # run.finish()

    return tokenizer, cap_vector, word_to_index, index_to_word
