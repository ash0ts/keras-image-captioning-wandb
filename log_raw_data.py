from dotenv import load_dotenv, find_dotenv
import os
import wandb
import tensorflow as tf
from config import WANDB_PROJECT, WANDB_ENTITY, annotation_folder, image_folder
load_dotenv(find_dotenv())


def log_raw_dataset():
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="log-coco2014", job_type="log_raw")
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath(
                                                     '.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)

        os.remove(annotation_zip)
        annotation_file = os.path.dirname(
            annotation_zip)+'/annotations/captions_train2014.json'
    else:
        annotation_file = './annotations/captions_train2014.json'

    # Download image files
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder

    raw_annot_art = wandb.Artifact("annotations", type="raw-data")
    raw_annot_art.add_file(annotation_file)
    run.log_artifact(raw_annot_art)

    raw_image_art = wandb.Artifact("images", type="raw-data")
    raw_image_art.add_dir(PATH)
    run.log_artifact(raw_image_art)
    run.finish()
    return None


if __name__ == "__main__":
    log_raw_dataset()
