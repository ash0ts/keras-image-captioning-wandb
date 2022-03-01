from model.CNNEncoder import CNN_Encoder
from model.RNNDecoder import RNN_Decoder
import tensorflow as tf
import time
from image_utils import load_image
import numpy as np
from config import WANDB_PROJECT, WANDB_ENTITY, max_length, vocabulary_size
import wandb
from utils import load_from_pickle
from text_utils import generate_text_artifacts
import os
# Feel free to change these parameters according to your system's configuration


def train():

    default_config = {
        "BATCH_SIZE": 64,
        "BUFFER_SIZE": 1000,
        "embedding_dim": 256,
        "units": 512,
        "features_shape": 2048,
        "attention_features_shape": 64
    }

    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY, name="train-coco2014-attention-model", job_type="train", config=default_config)

    BATCH_SIZE = run.config.get("BATCH_SIZE")
    BUFFER_SIZE = run.config.get("BUFFER_SIZE")
    embedding_dim = run.config.get("embedding_dim")
    units = run.config.get("units")
    features_shape = run.config.get("features_shape")
    attention_features_shape = run.config.get("attention_features_shape")

    split_art = run.use_artifact("split:latest")
    img_name_train_path = split_art.get_path("img_name_train.pkl").download()
    cap_train_path = split_art.get_path("cap_train.pkl").download()

    image_feat_art = run.use_artifact("inception_v3:latest")
    image_feat_path = image_feat_art.download()
    img_name_train = [os.path.join(image_feat_path, path)
                      for path in load_from_pickle(img_name_train_path)]
    cap_train = load_from_pickle(cap_train_path)

    # TODO: Very jank. We regenerate the tokenizer twice from these same data from a previous step which feels wrong. Hope this doesnt change.
    img_cap_table = run.use_artifact(
        "image_caption_table:latest").get("img_cap_table")
    train_captions = img_cap_table.get_column("caption")

    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)
    # cap vecotr contains each sentence as max_length where the word position index is the vocab index
    tokenizer, _, word_to_index, index_to_word = generate_text_artifacts(
        caption_dataset, max_length=max_length, vocabulary_size=vocabulary_size, return_vector=False, return_mapping=True)

    num_steps = len(img_name_train) // BATCH_SIZE
    # TODO: Load the numpy files from inception artifact

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

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    @ tf.function
    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims(
            [word_to_index('<start>')] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def grab_gradients(model):
        gradients = {}
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                gradients[layer.name] = layer.get_weights()[0]
        return gradients

    EPOCHS = 1

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            run.log({"batch_loss": batch_loss})
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                print(
                    f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        run.log({"epoch_loss": total_loss/num_steps})
        run.log({"encoder_epoch_gradient": grab_gradients(encoder)})
        run.log({"decoder_epoch_gradient": grab_gradients(decoder)})
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

    encoder.save("encoder")
    decoder.save("decoder")

    models = wandb.Artifact()
    run.finish()

    # plt.plot(loss_plot)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Plot')
    # plt.show()


if __name__ == "__main__":
    train()
