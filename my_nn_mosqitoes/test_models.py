from model import build_feature_extractor, build_twin_regressor, build_neural_network
from model2 import build_feature_extractor2
import json

data_folder = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"
with open(data_folder+"statistics.json") as data_stats_open:
            data_stats = json.load(data_stats_open)
            seq_length = data_stats['seq_length']
            target_length = data_stats['target_length']
fe_model, emb_units = build_feature_extractor( input_seq_len=seq_length, batch_size=2)
# fe_model, emb_units = build_feature_extractor2( input_seq_len=seq_length, batch_size=2)

print(fe_model.summary())
regressor_model = build_twin_regressor(united_embedding_size=emb_units, batch_size=2, target_size=target_length)
print(regressor_model.summary())

siamese_model, fe_layer, regression_model = build_neural_network(seq_length, 2, target_length)
print(siamese_model.summary())
checkpoint_path = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/checkpoint/20211115-112545checkpoint_model.h5"
# checkpoint_path = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/checkpoint/checkpoint"
siamese_model.load_weights(checkpoint_path)