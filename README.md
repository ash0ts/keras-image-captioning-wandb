# keras-image-captioning-wandb
 
> Based on [this image captioning tutorial by the Tensorflow Authors](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/image_captioning.ipynb)

The model architecture is similar to [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).

This notebook is an end-to-end example. When you run the notebook, it downloads the [MS-COCO](http://cocodataset.org/#home) dataset, preprocesses and caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

In this example, you will train a model on a relatively small amount of data—the first 30,000 captions  for about 20,000 images (because there are multiple captions per image in the dataset).

### 📚 Data
You will use the 2014 MS-COCO dataset to train your model. The dataset contains over 82,000 images, each of which has at least 5 different caption annotations

<ul> 
    <li> http://images.cocodataset.org/annotations/annotations_trainval2014.zip </li>
    <li> http://images.cocodataset.org/zips/train2014.zip </li>
</ul>



### ⏯ Commands
| Command | Description |
| --- | --- |
| `log_raw_data` | Use `tf.keras.utils.get_file` to download <ul> <li>http://images.cocodataset.org/annotations/annotations_trainval2014.zip <li>http://images.cocodataset.org/zips/train2014.zip </ul> Log these datasets to `wandb` AFTER unzipping |
| **current step**&rarr;`process_data` | Use InceptionV3 (which is pretrained on Imagenet) to extract features and cache the output to disk. Process and save the associated captions. Save the resultant prepared dataset that will be used to train/evaluate the model. Log these features to `wandb` in an Artifact and/or Table. <br/>**[TODO]**<br/> <ul><li>Remove hardcoded subsetting of the dataset <li>Log the inception features alongside the images in a `wandb.Table` for a nice embedding projector <li>Save and log processed captions. <li>Save and log input dataset for model training/eval </ul> |
| `train` | Pass config values and data to run a Keras training experiment using `WandbCallback` or such. [Model definition](./model) |
| `evaluate` | Calculate evaluation metrics, Generate a `wandb.Table` to yield visualizations for `Reports` such as viewing a plot of the attention map over the text generation from the model |
| `promote` | **OPTIONAL**: Use the above evaluation metrics to grab and promote the best model for production (similar to `model registry`) |

### ⏭ Workflows

| Workflow | Steps |
| --- | --- |
| `---` | `log_raw_data` &rarr; `process_data` &rarr; `train` &rarr; `evaluate` |