from dataset_generator import DatasetGen
import pandas as pd

data_folder = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"
seq_bed_data = pd.read_csv(data_folder+"sequences.bed", sep="\t",
names=["chr", "start", "end", "mode"])
datagen_for_training = DatasetGen(
    input_data_file=data_folder+"inputs",
    target_data_file=data_folder+"targets_5",
    data_stats_file=data_folder+"statistics.json",
    apply_augmentation=False,
    seq_data=seq_bed_data,
    mode="train",
    batch_size=10,
    batches_per_epoch=6,
    draw_target=True
)
print("n_samples", datagen_for_training.n_samples)
print("batches per epoch", datagen_for_training.batches_per_epoch)
datagen_for_training.__len__()
print("get item")
batch_input, batch_target, something = datagen_for_training.__getitem__(3)
print(batch_target)
print(batch_target.shape)