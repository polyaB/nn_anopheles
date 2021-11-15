import tensorflow as tf
import pandas as pd
import json
from dataset_generator import DatasetGen
from model import build_feature_extractor, build_twin_regressor, build_neural_network
import datetime

data_folder = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"
# data_folder = "/home/polina/nn_anopheles/test_new/"
log_dir = data_folder + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with open(data_folder+"statistics.json") as data_stats_open:
            data_stats = json.load(data_stats_open)
            seq_length = data_stats['seq_length']
            target_length = data_stats['target_length']
seq_bed_data = pd.read_csv(data_folder+"sequences.bed", sep="\t", names=["chr", "start", "end", "mode"])
batch_size = 2

siamese_model, fe_layer, regression_model = build_neural_network(seq_length, batch_size, target_length)

print(siamese_model.summary())
#write model summary in file
with open(log_dir+'modelsummary.txt', "w") as f:
  siamese_model.summary(print_fn=lambda x: f.write(x + '\n'))
  fe_layer.summary(print_fn=lambda x: f.write(x + '\n'))
  regression_model.summary(print_fn=lambda x: f.write(x + '\n'))


datagen_for_training = DatasetGen(
    input_data_file=data_folder+"inputs",
    target_data_file=data_folder+"targets_5",
    data_stats_file=data_folder+"statistics.json",
    apply_augmentation=False,
    seq_data=seq_bed_data,
    mode="train",
    batch_size=batch_size,
    batches_per_epoch=100,
)

datagen_for_validation = DatasetGen(
    input_data_file=data_folder+"inputs",
    target_data_file=data_folder+"targets_5",
    data_stats_file=data_folder+"statistics.json",
    apply_augmentation=False,
    seq_data=seq_bed_data,
    mode="valid",
    batch_size=batch_size,
    batches_per_epoch=None,
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_filepath = data_folder + 'checkpoint/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'checkpoint_model.h5'

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3000,
        verbose=True,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=False),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tf.keras.callbacks.TimeStopping(
    #     seconds=int(round(3600 * 1.9)),
    #     verbose=True
    # )
]
history = siamese_model.fit(datagen_for_training,
                                 validation_data=datagen_for_validation,
                                 epochs=10000,  callbacks=callbacks)