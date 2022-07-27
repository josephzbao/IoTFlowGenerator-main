import sys
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Concatenate
from keras.layers.embeddings import Embedding
from utilities import extract_packet_sizes, splitAllFeatures, extract_durations, extract_protocol_conf, extract_src_port, extract_dst_port, normalize_packet_sizes
import warnings
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import random

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

sys.setrecursionlimit(1000000)

warnings.filterwarnings('ignore')

# fix random seed for reproducibility
np.random.seed(7)

featuresFilePath = sys.argv[1]

with open(featuresFilePath, mode='rb') as featuresFile:
    raw_features = pickle.load(featuresFile)

device_to_packet_sizes = dict()
device_to_durations = dict()
device_to_protocol_conf = dict()
device_to_src_ports = dict()
device_to_dst_ports = dict()
raw_features = splitAllFeatures(raw_features)

all_keys = []

all_fake_features = dict()
all_real_features = dict()

packetinferenceresults = dict()
f1scores = dict()
f1normscores = dict()
mccscores = dict()

packet_size_min = 0
packet_size_max = 0

for key, value in raw_features.items():
    packet_size_sequences = extract_packet_sizes(value)
    for seq in packet_size_sequences:
        packet_size_min = min(min(seq), packet_size_min)
        packet_size_max = max(max(seq), packet_size_max)

protocol_conf_max = 0

for key, value in raw_features.items():
    protocol_conf_sequences = extract_protocol_conf(value)
    for seq in protocol_conf_sequences:
        protocol_conf_max = max(max(seq), protocol_conf_max)

src_port_max = 0

for key, value in raw_features.items():
    port_sequences = extract_src_port(value)
    for seq in port_sequences:
        src_port_max = max(max(seq), src_port_max)

dst_port_max = 0

for key, value in raw_features.items():
    port_sequences = extract_dst_port(value)
    for seq in port_sequences:
        dst_port_max = max(max(seq), dst_port_max)

for key, value in raw_features.items():
    all_keys.append(key)
    device_to_packet_sizes[key] = normalize_packet_sizes(extract_packet_sizes(value), max_packet_size=abs(packet_size_min))[0]
    durations = extract_durations(value)
    temp_durations = []
    for seq in durations:
        temp_dur = []
        for d in seq:
            temp_dur.append([d])
        temp_durations.append(temp_dur)
    device_to_durations[key] = extract_durations(value)
    device_to_protocol_conf[key] = extract_protocol_conf(value)
    device_to_src_ports[key] = extract_src_port(value)
    device_to_dst_ports[key] = extract_dst_port(value)
    device_to_protocol_conf[key] = extract_protocol_conf(value)

def baseline_model(packetTokenCount = packet_size_max + abs(packet_size_min) + 1, portsSrcCount = src_port_max, portsDstCount = dst_port_max, protocolCount = protocol_conf_max, n_classes=2, max_length=20):
    in_seqs = Input(shape=(max_length,))
    in_src_ports = Input(shape=(max_length,))
    in_dst_ports = Input(shape=(max_length,))
    in_protocols = Input(shape=(max_length,))
    in_durs = Input(shape=(max_length, 1))
    inp_s = Embedding(packetTokenCount + 1, 32, input_length=max_length)(in_seqs)
    inp_src_ports = Embedding(portsSrcCount + 1, 32, input_length=max_length)(in_src_ports)
    inp_dst_ports = Embedding(portsDstCount + 1, 32, input_length=max_length)(in_dst_ports)
    inp_protocols = Embedding(protocolCount + 1, 32, input_length=max_length)(in_protocols)
    merged = Concatenate(axis=2)([inp_s, in_durs, inp_src_ports, inp_dst_ports, inp_protocols])
    merged = LSTM(max_length)(merged)
    out1 = Dense(n_classes, activation='softmax')(merged)
    model = Model([in_seqs, in_durs, in_protocols, in_src_ports, in_dst_ports], out1)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

for key in all_keys:
    real = device_to_packet_sizes[key]
    real_durations = device_to_durations[key]
    real_protocol_confs = device_to_protocol_conf[key]
    real_src_ports = device_to_src_ports[key]
    real_dst_ports = device_to_dst_ports[key]
    fake = []
    fake_durations = []
    fake_protocl_confs = []
    fake_src_ports = []
    fake_dst_ports = []
    for a in all_keys:
        if a == key:
            continue
        fake += device_to_packet_sizes[a]
        fake_durations += device_to_durations[a]
        fake_protocl_confs += device_to_protocol_conf[a]
        fake_src_ports += device_to_src_ports[a]
        fake_dst_ports += device_to_dst_ports[a]
    test_split_factor = 0.33
    total_fakes_needed = len(fake) * test_split_factor
    if len(real) * 2 < 100:
        extra_fake_features, fake, extra_fake_durations, fake_durations, extra_fake_protocol_confs, fake_protocl_confs, extra_fake_src_ports, fake_src_ports, extra_fake_dst_ports, fake_dst_ports = train_test_split(fake, fake_durations, fake_protocl_confs, fake_src_ports, fake_dst_ports, test_size=100 / len(fake), random_state=42)
        if len(fake) * test_split_factor < total_fakes_needed:
            extra_fake_idx = random.sample(range(0, len(extra_fake_features)), int(total_fakes_needed - (len(fake) * test_split_factor)))
            extra_fake_features_temp = []
            extra_fake_durations_temp = []
            extra_fake_protocol_confs_temp = []
            extra_fake_src_ports_temp = []
            extra_fake_dst_ports_temp = []
            for idx in extra_fake_idx:
                extra_fake_features_temp.append(extra_fake_features[idx])
                extra_fake_durations_temp.append(extra_fake_durations[idx])
                extra_fake_protocol_confs_temp.append(extra_fake_protocol_confs[idx])
                extra_fake_src_ports_temp.append(extra_fake_src_ports[idx])
                extra_fake_dst_ports_temp.append(extra_fake_dst_ports[idx])
            extra_fake_features = extra_fake_features_temp
            extra_fake_durations = extra_fake_durations_temp
            extra_fake_protocol_confs = extra_fake_protocol_confs_temp
            extra_fake_src_ports = extra_fake_src_ports_temp
            extra_fake_dst_ports = extra_fake_dst_ports_temp
        else:
            extra_fake_features = []
            extra_fake_durations = []
            extra_fake_protocol_confs = []
            extra_fake_src_ports = []
            extra_fake_dst_ports = []
    else:
        if len(real) * 2 < len(fake):
            extra_fake_features, fake, extra_fake_durations, fake_durations, extra_fake_protocol_confs, fake_protocl_confs, extra_fake_src_ports, fake_src_ports, extra_fake_dst_ports, fake_dst_ports = train_test_split(
                fake, fake_durations, fake_protocl_confs, fake_src_ports, fake_dst_ports, test_size= len(real) * 2 / len(fake),
                random_state=42)
            if len(fake) * test_split_factor < total_fakes_needed:
                extra_fake_idx = random.sample(range(0, len(extra_fake_features)), int(total_fakes_needed - (len(fake) * test_split_factor)))
                extra_fake_features_temp = []
                extra_fake_durations_temp = []
                extra_fake_protocol_confs_temp = []
                extra_fake_src_ports_temp = []
                extra_fake_dst_ports_temp = []
                for idx in extra_fake_idx:
                    extra_fake_features_temp.append(extra_fake_features[idx])
                    extra_fake_durations_temp.append(extra_fake_durations[idx])
                    extra_fake_protocol_confs_temp.append(extra_fake_protocol_confs[idx])
                    extra_fake_src_ports_temp.append(extra_fake_src_ports[idx])
                    extra_fake_dst_ports_temp.append(extra_fake_dst_ports[idx])
                extra_fake_features = extra_fake_features_temp
                extra_fake_durations = extra_fake_durations_temp
                extra_fake_protocol_confs = extra_fake_protocol_confs_temp
                extra_fake_src_ports = extra_fake_src_ports_temp
                extra_fake_dst_ports = extra_fake_dst_ports_temp
            else:
                extra_fake_features = []
                extra_fake_durations = []
                extra_fake_protocol_confs = []
                extra_fake_src_ports = []
                extra_fake_src_ports = []
        else:
            extra_fake_features = []
            extra_fake_durations = []
            extra_fake_protocol_confs = []
            extra_fake_src_ports = []
            extra_fake_src_ports = []

    labels = np.array([0] * (len(real)) + ([1] * len(fake)))
    features = np.array(real + fake)
    features_protocol_confs = np.array(real_protocol_confs + fake_protocl_confs)
    features_src_ports = np.array(real_src_ports + fake_src_ports)
    features_dst_ports = np.array(real_dst_ports + fake_dst_ports)
    features_durations = np.array(real_durations + fake_durations)
    X_train, X_test, X_train_dur, X_test_dur, X_train_proto, X_test_proto, X_train_src_ports, X_test_src_ports, X_train_dst_ports, X_test_dst_ports, y_train, y_test = train_test_split(features, features_durations, features_protocol_confs, features_src_ports, features_dst_ports, labels, test_size=test_split_factor, random_state=42)
    X_test = np.array(list(X_test) + extra_fake_features)
    X_test_dur = np.array(list(X_test_dur) + extra_fake_durations)
    X_test_proto = np.array(list(X_test_proto) + extra_fake_protocol_confs_temp)
    X_test_src_ports = np.array(list(X_test_src_ports) + extra_fake_src_ports)
    X_test_dst_ports = np.array(list(X_test_dst_ports) + extra_fake_dst_ports)
    y_test = np.array(list(y_test) + ([1] * len(extra_fake_features)))
    discriminator = baseline_model()
    discriminator.fit([X_train, X_train_dur, X_train_proto, X_train_src_ports, X_train_dst_ports], y_train, epochs=500, batch_size=16, verbose=1)
    discriminator.summary()
    plot_model(discriminator, to_file='packetinfoadversarynn.png', show_shapes=True, show_layer_names=True)
    predictions = discriminator.predict([X_test, X_test_dur, X_test_proto, X_test_src_ports, X_test_dst_ports])
    correct = 0
    wrong = 0
    predicted = []
    positives = 0
    negatives = 0

    for i in range(len(predictions)):
        predicted_class = np.argmax(predictions[i])
        predicted.append(predicted_class)
        target_class = y_test[i]
        if target_class == 0:
            positives += 1
        else:
            negatives += 1
        if predicted_class == target_class:
            correct += 1
        else:
            wrong += 1

    accuracy = correct / (correct + wrong)
    print(key)
    print("Accuracy of model is ")
    print(accuracy)
    packetinferenceresults[key] = accuracy
    results = confusion_matrix(y_test, predicted)
    print(results)
    print("f1 score")
    f1 = f1_score(y_test, predicted, pos_label=0, average='binary')
    print(f1)
    f1scores[key] = f1
    q = positives / (positives + negatives)
    f1_coin = (2 * q) / (q + 1)
    f1_norm = (f1 - f1_coin) / (1 - f1_coin)
    print("f1 norm")
    print(f1_norm)
    f1normscores[key] = f1_norm

    mcc = matthews_corrcoef(y_test, predicted)
    print("mcc")
    print(mcc)
    mccscores[key] = mcc

with open("packetsizedurationinferenceresultsexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(packetinferenceresults, featureOutputFile)

with open("packetsizedurationf1experiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(f1scores, featureOutputFile)

with open("packetsizedurationf1normexperiment0.pkl", mode='wb') as featureOutputFile:
    pickle.dump(f1normscores, featureOutputFile)
