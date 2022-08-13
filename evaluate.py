from keras import Sequential
from keras.models import Model, Input
from sklearn.metrics import accuracy_score, f1_score

from preprocess import protocol_list
from keras.layers import LSTM, Embedding, Dense, Concatenate
from embed import ngrams, matches, extractDuration, signatureExtractionAll, filterNoisy, \
    extractPacketFrameLengthsWithDirection
import numpy as np
from itertools import groupby
import statistics
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from statistics import multimode
import glob
import pickle
import os

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

def evaluate_directory(directory, duration_cluster_size, n_embeddings, embedding_dim):
    extended = directory + "/*"
    paths = glob.glob(extended)
    all_devices_all_real = dict()
    all_devices_all_fake = dict()
    all_devices_all_signatures = dict()
    for path in paths:
        print(path)
        directoryStr = str(path[len(directory) + 1:])
        with open(path + "/" + duration_cluster_size + "/" + n_embeddings + "/" + embedding_dim + "/fake_device_features.pkl", mode='rb') as pklfile:
            all_devices_all_fake[directoryStr] = pickle.load(pklfile)
            pklfile.close()
        with open(path + "/" + duration_cluster_size + "/" + n_embeddings + "/" + embedding_dim + "/real_device_features.pkl", mode='rb') as pklfile:
            all_devices_all_real[directoryStr] = pickle.load(pklfile)
            pklfile.close()
        with open(path + "/all_signatures.pkl", mode='rb') as pklfile:
            all_devices_all_signatures[directoryStr] = pickle.load(pklfile)
            pklfile.close()
    packetInfoResults, trafficRateResults, signatureFrequencyResults = evaluate_all_devices(all_devices_all_real, all_devices_all_fake, all_devices_all_signatures)
    os.makedirs("/home/joseph/vqvae/results/" + duration_cluster_size + "/" + n_embeddings + "/" + embedding_dim, exist_ok=True)
    with open("/home/joseph/vqvae/results/" + duration_cluster_size + "/" + n_embeddings + "/" + embedding_dim + "/packetInfoResults.pkl", mode='wb+') as pklfile:
        pickle.dump(packetInfoResults, pklfile)
        pklfile.close()
    with open("/home/joseph/vqvae/results/" + duration_cluster_size + "/" + n_embeddings + "/" + embedding_dim + "/trafficRateResults.pkl", mode='wb+') as pklfile:
        pickle.dump(trafficRateResults, pklfile)
        pklfile.close()
    with open("/home/joseph/vqvae/results/" + duration_cluster_size + "/" + n_embeddings + "/" + embedding_dim + "/signatureFrequencyResults.pkl", mode='wb+') as pklfile:
        pickle.dump(signatureFrequencyResults, pklfile)
        pklfile.close()
    return packetInfoResults, trafficRateResults, signatureFrequencyResults

def evaluate_all_devices(all_devices_all_real, all_devices_all_fake, all_devices_all_signatures):
    signatureFrequencyResults = dict()
    for device in all_devices_all_real.keys():
        signatureFrequencyResults[device] = evaluate_for_device(device, all_devices_all_real, all_devices_all_fake, all_devices_all_signatures[device])
    packetInfoResults, trafficRateResults = evaluate(all_devices_all_real, all_devices_all_fake)
    return packetInfoResults, trafficRateResults, signatureFrequencyResults

def evaluate_for_device(device, all_devices_all_real, all_devices_all_fake, signaturesForDevice):
    all_real_signature_frequency = dict()
    all_fake_signature_frequency = dict()
    total_signatures = sum([len(signaturesForDevice[item]) for item in signaturesForDevice.keys()])
    for key, value in all_devices_all_real.items():
        trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, value))
        frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
        if len(np.array(frameLengthsAndDirection).flatten()) == 0 or not any(frameLengthsAndDirection):
            all_real_signature_frequency[key] = len(value) * [[0] * total_signatures]
        else:
            all_real_signature_frequency[key] = filterOnLength(
                preprocessForSignatureFrequency(frameLengthsAndDirection, signaturesForDevice), total_signatures)
    for key, value in all_devices_all_fake.items():
        trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, value))
        frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
        if len(np.array(frameLengthsAndDirection).flatten()) == 0 or not any(frameLengthsAndDirection):
            all_fake_signature_frequency[key] = len(value) * [[0] * total_signatures]
        else:
            all_fake_signature_frequency[key] = filterOnLength(preprocessForSignatureFrequency(frameLengthsAndDirection,
                                                                                               signaturesForDevice),
                                                               total_signatures)
    if all_real_signature_frequency[device] is not None and all_fake_signature_frequency[device] is not None:
        if len(all_real_signature_frequency[device]) < 2 or len(all_fake_signature_frequency[device]) < 2:
            return None
        else:
            return evaluateForSignatureFrequency(all_real_signature_frequency[device], randomlySelect(all_real_signature_frequency, device, len(
                                                                                   all_real_signature_frequency[device])),
                                                                           all_fake_signature_frequency[device],
                                                                           randomlySelect(all_fake_signature_frequency,
                                                                                          device, len(
                                                                                   all_fake_signature_frequency[device])))
            # signatureFrequencyResults[key] = evaluateForSignatureFrequency(all_real_signature_frequency[key], all_real_signature_frequency[key], all_fake_signature_frequency[key], all_fake_signature_frequency[key])
    else:
        return None

def evaluate(all_real, all_fake):
    max_duration = getDurationMax(all_real)
    packet_frame_max = getPacketFrameMax(all_real)
    frame_length_to_token = getPacketFrameLengthToToken(all_real, all_fake)
    src_port_to_token = getSrcPortToToken(all_real, all_fake)
    dst_port_to_token = getDstPortToToken(all_real, all_fake)
    print("start")
    print(frame_length_to_token)
    print(src_port_to_token)
    print(dst_port_to_token)
    all_real_packet_info = dict()
    all_real_traffic_rate = dict()
    all_fake_packet_info = dict()
    all_fake_traffic_rate = dict()
    packetInfoResults = dict()
    trafficRateResults = dict()
    for key, value in all_real.items():
        all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports = preprocessForPacketInfo(
            value, frame_length_to_token, src_port_to_token, dst_port_to_token, packet_frame_max,
            max_duration)
        all_real_packet_info[key] = [all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports]
        all_real_traffic_rate[key] = preprocessForTrafficRate(value)
    for key, value in all_fake.items():
        all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports = preprocessForPacketInfo(value, frame_length_to_token, src_port_to_token, dst_port_to_token, packet_frame_max,
                                max_duration)
        all_fake_packet_info[key] = [all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports]
        all_fake_traffic_rate[key] = preprocessForTrafficRate(value)
    for key in all_real.keys():
        packetInfoResults[key] = evaluateForPacketInfo(all_real_packet_info[key], randomlySelectAll(all_real_packet_info, key, len(all_real_packet_info[key][0])), all_fake_packet_info[key], randomlySelectAll(all_fake_packet_info, key, len(all_fake_packet_info[key][0])), len(frame_length_to_token), len(src_port_to_token), len(dst_port_to_token))
        trafficRateResults[key] = evaluateForTrafficRate(all_real_traffic_rate[key], randomlySelect(all_real_traffic_rate, key, len(all_real_traffic_rate[key])), all_fake_traffic_rate[key], randomlySelect(all_fake_traffic_rate, key, len(all_fake_traffic_rate[key])))
    return packetInfoResults, trafficRateResults

# def evaluater(all_real, all_fake, all_signatures):
#     max_duration = getDurationMax(all_real)
#     packet_frame_max = getPacketFrameMax(all_real)
#     frame_length_to_token = getPacketFrameLengthToToken(all_real)
#     src_port_to_token = getSrcPortToToken(all_real)
#     dst_port_to_token = getDstPortToToken(all_real)
#     all_real_packet_info = dict()
#     all_real_traffic_rate = dict()
#     all_real_signature_frequency = dict()
#     all_fake_packet_info = dict()
#     all_fake_traffic_rate = dict()
#     all_fake_signature_frequency = dict()
#     packetInfoResults = dict()
#     trafficRateResults = dict()
#     signatureFrequencyResults = dict()
#     for key, value in all_real.items():
#         all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports = preprocessForPacketInfo(
#             value, frame_length_to_token, src_port_to_token, dst_port_to_token, packet_frame_max,
#             max_duration)
#         all_real_packet_info[key] = [all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports]
#         all_real_traffic_rate[key] = preprocessForTrafficRate(value)
#         trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, value))
#         frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
#         signaturesForKey = all_signatures[key]
#         total_signatures = sum([len(signaturesForKey[item]) for item in signaturesForKey.keys()])
#         if len(np.array(frameLengthsAndDirection).flatten()) == 0 or not any(frameLengthsAndDirection):
#             all_real_signature_frequency[key] = len(all_directions) * [[0] * total_signatures]
#         else:
#             all_real_signature_frequency[key] = filterOnLength(preprocessForSignatureFrequency(frameLengthsAndDirection, signaturesForKey), total_signatures)
#     for key, value in all_fake.items():
#         all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports = preprocessForPacketInfo(value, frame_length_to_token, src_port_to_token, dst_port_to_token, packet_frame_max,
#                                 max_duration)
#         all_fake_packet_info[key] = [all_packet_frame_length_tokens, all_packet_frame_lengths_continuous, all_directions, all_durs, all_protocols, all_src_ports, all_dst_ports]
#         all_fake_traffic_rate[key] = preprocessForTrafficRate(value)
#         trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, value))
#         frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
#         if len(np.array(frameLengthsAndDirection).flatten()) == 0 or not any(frameLengthsAndDirection):
#             all_fake_signature_frequency[key] = len(all_directions) * [[0] * total_signatures]
#         else:
#             all_fake_signature_frequency[key] = filterOnLength(preprocessForSignatureFrequency(frameLengthsAndDirection,
#                                                                                 all_signatures), total_signatures)
#     for key in all_real.keys():
#         packetInfoResults[key] = evaluateForPacketInfo(all_real_packet_info[key], randomlySelectAll(all_real_packet_info, key, len(all_real_packet_info[key][0])), all_fake_packet_info[key], randomlySelectAll(all_fake_packet_info, key, len(all_fake_packet_info[key][0])), len(frame_length_to_token), len(src_port_to_token), len(dst_port_to_token))
#         trafficRateResults[key] = evaluateForTrafficRate(all_real_traffic_rate[key], randomlySelect(all_real_traffic_rate, key, len(all_real_traffic_rate[key])), all_fake_traffic_rate[key], randomlySelect(all_fake_traffic_rate, key, len(all_fake_traffic_rate[key])))
#         if all_real_signature_frequency[key] is not None and all_fake_signature_frequency[key] is not None:
#             if len(all_real_signature_frequency[key]) < 2 or len(all_fake_signature_frequency[key]) < 2:
#                 signatureFrequencyResults[key] = None
#             else:
#                 signatureFrequencyResults[key] = evaluateForSignatureFrequency(all_real_signature_frequency[key], randomlySelect(all_real_signature_frequency, key, len(all_real_signature_frequency[key])), all_fake_signature_frequency[key], randomlySelect(all_fake_signature_frequency, key, len(all_fake_signature_frequency[key])))
#                 # signatureFrequencyResults[key] = evaluateForSignatureFrequency(all_real_signature_frequency[key], all_real_signature_frequency[key], all_fake_signature_frequency[key], all_fake_signature_frequency[key])
#         else:
#             signatureFrequencyResults[key] = None
#     return packetInfoResults, trafficRateResults, signatureFrequencyResults

def filterOnLength(seqs, n):
    return list(filter(lambda x: len(x) == n, seqs))

def randomlySelectAll(all, avoidKey, n):
    pool = []
    for key, value in all.items():
        if key != avoidKey:
            for i in range(len(value[0])):
                pool.append((key, i))
    indexChoices = random.choices(pool, k=n)
    results = []
    for i in range(len(list(all.values())[0])):
        result = []
        for choice in indexChoices:
            result.append(all[choice[0]][i][choice[1]])
        results.append(result)
    return results

def randomlySelect(all, avoidKey, n):
    pool = []
    for key, value in all.items():
        if key != avoidKey:
            pool += value
    return random.choices(pool, k=n)

def getPacketFrames(all_real):
    all_packet_frame_lengths = []
    for device_features in all_real.values():
        for sequence in device_features:
            packet_frame_sequence = []
            for row in sequence:
                if row[0] is True:
                    store = row[1] * (1 if row[2] == True else -1)
                    packet_frame_sequence.append(store)
            all_packet_frame_lengths.append(packet_frame_sequence)
    return all_packet_frame_lengths

def getDurationMax(all_real):
    print(all_real)
    all_duration_max = []
    for device_features in all_real.values():
        for sequence in device_features:
            for row in sequence:
                all_duration_max.append(row[3])
    return max(all_duration_max)

def getPacketFrameMax(all_real):
    all_packet_frame_lengths = []
    for device_features in all_real.values():
        for sequence in device_features:
            for row in sequence:
                all_packet_frame_lengths.append(row[1])
    return max(all_packet_frame_lengths)

def getPacketFrameLengthToToken(all_real, all_fake):
    all_packet_frame_lengths = []
    for device_features in all_real.values():
        for sequence in device_features:
            for row in sequence:
                all_packet_frame_lengths.append(row[1])
    for device_features in all_fake.values():
        for sequence in device_features:
            for row in sequence:
                all_packet_frame_lengths.append(row[1])
    frameLengthToToken = dict()
    counter = 0
    for token in set(all_packet_frame_lengths):
        frameLengthToToken[token] = counter
        counter += 1
    return frameLengthToToken

def getSrcPortToToken(all_real, all_fake):
    all_src_ports = []
    for device_features in all_real.values():
        for sequence in device_features:
            for row in sequence:
                all_src_ports.append(row[5])
    for device_features in all_fake.values():
        for sequence in device_features:
            for row in sequence:
                all_src_ports.append(row[5])
    srcPortToToken = dict()
    counter = 0
    for token in set(all_src_ports):
        srcPortToToken[token] = counter
        counter += 1
    return srcPortToToken

def getDstPortToToken(all_real, all_fake):
    all_dst_ports = []
    for device_features in all_real.values():
        for sequence in device_features:
            for row in sequence:
                all_dst_ports.append(row[6])
    for device_features in all_fake.values():
        for sequence in device_features:
            for row in sequence:
                all_dst_ports.append(row[6])
    dstPortToToken = dict()
    counter = 0
    for token in set(all_dst_ports):
        dstPortToToken[token] = counter
        counter += 1
    return dstPortToToken

def group_then_convert(interval, packet_tuples):
    groups_iter = groupby(packet_tuples, lambda x: int(x[1]/interval))
    groups = []
    for key, group in groups_iter:
        group_array = []
        for thing in group:
            group_array.append(thing[0])
        groups.append((key, group_array))
    return groups

def group(interval, packet_tuples):
    groups_iter = groupby(packet_tuples, lambda x: int(x[1]/interval))
    groups = []
    for key, group in groups_iter:
        group_array = []
        for thing in group:
            group_array.append(thing)
        groups.append((key, group_array))
    return groups

def mean(lst):
    return sum(lst) / len(lst)

def durationsToTimestamp(all_durations, max_duration=1.0):
    all_timestamps = []
    for durations in all_durations:
        timestamps = []
        float_durations = [float(x) for x in durations]
        for i in range(len(float_durations)):
            timestamps.append(sum(float_durations[0:i+1]) * max_duration)
        all_timestamps.append(timestamps)
    return all_timestamps

def concat_all(all_packets, all_durations):
    result = []
    for i in range(len(all_packets)):
        packets = all_packets[i]
        durations = all_durations[i]
        if len(packets) != len(durations):
            break
        result.append(concat(packets, durations))
    return result

def concat(packets, durations):
    result = []
    for i in range(len(packets)):
        packet_size = packets[i]
        duration = durations[i]
        result.append((packet_size, duration))
    return result

def extractPacketFrameLengths(trafficFlowFeatures):
    return list(map(keepPacketFrameLengths, trafficFlowFeatures))

def keepPacketFrameLengths(trafficFlow):
    return list(map(lambda x: x[1], trafficFlow))

# find total traffic output through secondInterval, compute mean/std on firstInterval, create a feature vector with mean of each previous mean and std
def generate_traffic_rate_features(tuples, firstInterval = 10.0, secondIntervals = [0.1]):
    features = []
    for sequence in tuples:
        intervals = group(firstInterval, sequence)
        means = []
        stds = []
        fv = []
        for j, k in intervals:
            keyStream = k
            for secondInterval in secondIntervals:
                sub_groups = group_then_convert(secondInterval, keyStream)
                sub_group = []
                for k, v in sub_groups:
                    float_v = [abs(float(x)) for x in v]
                    sub_group.append(sum(float_v))
                sub_group = sub_group + ([0] * (int((firstInterval)/(secondInterval)) - len(sub_group)))
                means.append(mean(sub_group))
                stds.append(statistics.stdev(sub_group))
                fv.append([mean(sub_group), statistics.stdev(sub_group)])
        features.append(fv)
    return features

def separate(all_real, all_fake, target_device):
    real_target = []
    real_other = []
    fake_target = []
    fake_other = []
    for device, features in all_real.items():
        if device == target_device:
            real_target += features
        else:
            real_other += features
    for device, features in all_fake.items():
        if device == target_device:
            fake_target += features
        else:
            fake_other += features
    return real_target, real_other, fake_target, fake_other

def preprocessForPacketInfo(sequences, packetFrameLengthToToken, srcPortToToken, dstPortToToken, frameMax, durationMax):
    all_packet_frame_length_tokens = []
    all_packet_frame_lengths = []
    all_directions = []
    all_src_ports = []
    all_dst_ports = []
    all_protocols = []
    all_durs = []
    for sequence in sequences:
        packet_frame_length_tokens = []
        packet_frame_lengths = []
        directions = []
        src_ports = []
        dst_ports = []
        protocols = []
        durs = []
        for row in sequence:
            packet_frame_length_tokens.append([packetFrameLengthToToken[row[1]]])
            packet_frame_lengths.append([row[1]])
            directions.append([1] if row[2] == True else [0])
            durs.append([float(row[3]/durationMax)])
            protocols.append(row[4])
            if row[5] == 0:
                src_ports.append([0])
            else:
                src_ports.append([srcPortToToken[row[5]]])
            if row[6] == 0:
                dst_ports.append([0])
            else:
                dst_ports.append([dstPortToToken[row[6]]])
        all_packet_frame_length_tokens.append(packet_frame_length_tokens)
        all_packet_frame_lengths.append(np.array(packet_frame_lengths))
        all_directions.append(directions)
        all_durs.append(durs)
        all_protocols.append(protocols)
        all_src_ports.append(np.array(src_ports))
        all_dst_ports.append(np.array(dst_ports))
    all_packet_frame_lengths_continuous = np.array(all_packet_frame_lengths) / frameMax
    return np.array(all_packet_frame_length_tokens), all_packet_frame_lengths_continuous, np.array(all_directions), np.array(all_src_ports), np.array(all_dst_ports), np.array(all_protocols), np.array(all_durs)

def preprocessForSignatureFrequency(sequences, all_signatures):
    all_features = []
    for sequence in sequences:
        features = []
        for nGramLength, signatures in all_signatures.items():
            denominator = len(sequence) - nGramLength
            if denominator <= 0:
                features += len(signatures) * [0.0]
                continue;
            for signature in signatures:
                count = 0
                ngramsInSequence = ngrams(nGramLength, sequence)
                for ngram in ngramsInSequence:
                    if matches(ngram, signature):
                        count += 1
                features.append(abs(float(count)/float(denominator)))
        all_features.append(features)
    return np.array(all_features)

def preprocessForTrafficRate(sequences):
    return generate_traffic_rate_features(concat_all(extractPacketFrameLengths(sequences), durationsToTimestamp(extractDuration(sequences))))

def fitModel(X_train, y_train):
    X_train_final = []
    y_train_final = []
    for i in range(len(X_train)):
        X_train_row = X_train[i]
        y_train_row = y_train[i]
        for t in X_train_row:
            X_train_final.append(t)
            y_train_final.append(y_train_row)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(np.array(X_train_final), np.array(y_train_final))
    return neigh

def predictWith(model, X_test):
    predictions = []
    for row in X_test:
        prediction = model.predict(row)
        predictions.append(multimode(prediction)[0])
    return np.array(predictions)

def evaluateForTrafficRate(real_target, real_other, fake_target, fake_other):
    experiments = dict()

    # Experiment 0 Train real, test real
    experimentNumber = 0
    features = real_target + real_other
    labels = [1] * len(real_target) + [0] * len(real_other)
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size = 0.33, random_state = 42)
    fittedModel = fitModel(X_train, y_train)
    y_predicted = predictWith(fittedModel, X_test)
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 1 Train real, test fake
    experimentNumber = 1
    X_train = np.array(real_target + real_other)
    y_train = np.array([1] * len(real_target) + [0] * len(real_other))
    X_test = np.array(fake_target + fake_other)
    y_test = np.array([1] * len(fake_target) + [0] * len(fake_other))
    fittedModel = fitModel(X_train, y_train)
    y_predicted = predictWith(fittedModel, X_test)
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 2 Train fake, test real
    experimentNumber = 2
    X_train = np.array(fake_target + fake_other)
    y_train = np.array([1] * len(fake_target) + [0] * len(fake_other))
    X_test = np.array(real_target + real_other)
    y_test = np.array([1] * len(real_target) + [0] * len(real_other))
    fittedModel = fitModel(X_train, y_train)
    y_predicted = predictWith(fittedModel, X_test)
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 3 Train both, test both
    experimentNumber = 3
    features = np.array(real_target + fake_target)
    labels = np.array([1] * len(real_target) + [0] * len(fake_target))
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.33,
                                                        random_state=42)
    fittedModel = fitModel(X_train, y_train)
    y_predicted = predictWith(fittedModel, X_test)
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    return experiments

def evaluateForPacketInfo(real_target, real_other, fake_target, fake_other, numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts):
    experiments = dict()

    # Experiment 0 Train real, test real
    experimentNumber = 0

    features_0 = np.concatenate((np.array(real_target[0]), np.array(real_other[0])), axis=0)
    features_1 = np.concatenate((np.array(real_target[1]), np.array(real_other[1])), axis=0)
    features_2 = np.concatenate((np.array(real_target[2]), np.array(real_other[2])), axis=0)
    features_3 = np.concatenate((np.array(real_target[3]), np.array(real_other[3])), axis=0)
    features_4 = np.concatenate((np.array(real_target[4]), np.array(real_other[4])), axis=0)
    features_5 = np.concatenate((np.array(real_target[5]), np.array(real_other[5])), axis=0)
    features_6 = np.concatenate((np.array(real_target[6]), np.array(real_other[6])), axis=0)
    labels = [1] * len(real_target[0]) + [0] * len(real_other[0])
    X_train_0, X_test_0, X_train_1, X_test_1, X_train_2, X_test_2, X_train_3, X_test_3, X_train_4, X_test_4, X_train_5, X_test_5, X_train_6, X_test_6, y_train, y_test = train_test_split(np.array(features_0), np.array(features_1), np.array(features_2), np.array(features_3), np.array(features_4), np.array(features_5), np.array(features_6), np.array(labels), test_size = 0.33, random_state = 42)
    print(numberOfPacketFrames)
    print(numberOfSrcPorts)
    print(numberOfDstPorts)
    model = packetInfoModel(numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts)
    model.fit([X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6], y_train, batch_size=32, epochs=30)
    y_predicted = model.predict([X_test_0, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6], verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 1 Train real, test fake
    experimentNumber = 1
    y_train = np.array(([1] * len(real_target[0])) + ([0] * len(real_other[0])))
    fake_features_0 = np.concatenate((np.array(fake_target[0]), np.array(fake_other[0])), axis=0)
    fake_features_1 = np.concatenate((np.array(fake_target[1]), np.array(fake_other[1])), axis=0)
    fake_features_2 = np.concatenate((np.array(fake_target[2]), np.array(fake_other[2])), axis=0)
    fake_features_3 = np.concatenate((np.array(fake_target[3]), np.array(fake_other[3])), axis=0)
    fake_features_4 = np.concatenate((np.array(fake_target[4]), np.array(fake_other[4])), axis=0)
    fake_features_5 = np.concatenate((np.array(fake_target[5]), np.array(fake_other[5])), axis=0)
    fake_features_6 = np.concatenate((np.array(fake_target[6]), np.array(fake_other[6])), axis=0)
    y_test = np.array([1] * len(fake_target[0]) + [0] * len(fake_other[0]))
    model = packetInfoModel(numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts)
    model.fit([features_0, features_1, features_2, features_3, features_4, features_5, features_6], y_train, batch_size=32, epochs=30)
    y_predicted = model.predict([fake_features_0, fake_features_1, fake_features_2, fake_features_3, fake_features_4, fake_features_5, fake_features_6], verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 2 Train fake, test real
    experimentNumber = 2
    y_train = np.array(([1] * len(fake_target[0])) + ([0] * len(fake_other[0])))
    y_test = np.array([1] * len(real_target[0]) + [0] * len(real_other[0]))
    model = packetInfoModel(numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts)
    model.fit([fake_features_0, fake_features_1, fake_features_2, fake_features_3, fake_features_4, fake_features_5, fake_features_6], y_train, batch_size=32, epochs=30)
    y_predicted = model.predict([features_0, features_1, features_2, features_3, features_4, features_5, features_6], verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 3 Train both, test both
    experimentNumber = 3
    features_0 = np.concatenate((np.array(real_target[0]), np.array(fake_target[0])), axis=0)
    features_1 = np.concatenate((np.array(real_target[1]), np.array(fake_target[1])), axis=0)
    features_2 = np.concatenate((np.array(real_target[2]), np.array(fake_target[2])), axis=0)
    features_3 = np.concatenate((np.array(real_target[3]), np.array(fake_target[3])), axis=0)
    features_4 = np.concatenate((np.array(real_target[4]), np.array(fake_target[4])), axis=0)
    features_5 = np.concatenate((np.array(real_target[5]), np.array(fake_target[5])), axis=0)
    features_6 = np.concatenate((np.array(real_target[6]), np.array(fake_target[6])), axis=0)
    labels = np.array([1] * len(real_target[0]) + [0] * len(fake_target[0]))
    X_train_0, X_test_0, X_train_1, X_test_1, X_train_2, X_test_2, X_train_3, X_test_3, X_train_4, X_test_4, X_train_5, X_test_5, X_train_6, X_test_6, y_train, y_test = train_test_split(np.array(features_0), np.array(features_1), np.array(features_2), np.array(features_3), np.array(features_4), np.array(features_5), np.array(features_6), np.array(labels), test_size = 0.33, random_state = 42)
    model = packetInfoModel(numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts)
    model.fit([X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6], y_train, batch_size=32, epochs=50)
    y_predicted = model.predict([X_test_0, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6], verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    return experiments

def evaluateForSignatureFrequency(real_target, real_other, fake_target, fake_other):
    experiments = dict()

    # Experiment 0 Train real, test real
    experimentNumber = 0
    features = np.concatenate((np.array(real_target), np.array(real_other)), axis=0)
    labels = [1] * len(real_target) + [0] * len(real_other)
    X_train, X_test, y_train, y_test = train_test_split(features, np.array(labels), test_size=0.33,
                                                        random_state=42)
    model = signature_frequency_model()
    model.fit(X_train, y_train, batch_size=32, epochs=30)
    y_predicted = model.predict(X_test, verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 1 Train real, test fake
    experimentNumber = 1
    X_test = np.concatenate((np.array(fake_target), np.array(fake_other)), axis=0)
    y_test = np.array([1] * len(fake_target) + [0] * len(fake_other))
    model = signature_frequency_model()
    model.fit(features, np.array(labels), batch_size=32, epochs=30)
    y_predicted = model.predict(X_test, verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 2 Train fake, test real
    experimentNumber = 2
    X_train = np.concatenate((np.array(fake_target), np.array(fake_other)), axis=0)
    y_train = np.array([1] * len(fake_target) + [0] * len(fake_other))
    X_test = np.concatenate((np.array(real_target), np.array(real_other)), axis=0)
    y_test = np.array([1] * len(real_target) + [0] * len(real_other))
    model = signature_frequency_model()
    model.fit(X_train, y_train, batch_size=32, epochs=30)
    y_predicted = model.predict(X_test, verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    # Experiment 3 Train both, test both
    experimentNumber = 3
    features = np.concatenate((np.array(real_target), np.array(fake_target)), axis=0)
    labels = np.array([1] * len(real_target) + [0] * len(fake_target))
    X_train, X_test, y_train, y_test = train_test_split(features, np.array(labels), test_size=0.33,
                                                        random_state=42)
    model = signature_frequency_model()
    model.fit(X_train, y_train, batch_size=32, epochs=30)
    y_predicted = model.predict(X_test, verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    metrics_dict = {"accuracy": accuracy_score(y_test, y_predicted), "f1": f1_score(y_test, y_predicted)}
    experiments[experimentNumber] = metrics_dict

    return experiments

def swap(current_indices, all_indices, x):
    sampled = []
    while len(sampled) < x:
        chosen = random.choice(all_indices)
        if chosen not in current_indices:
            sampled.append(chosen)
    for i in range(len(sampled)):
        el = random.sample(current_indices, 1)[0]
        current_indices.remove(el)
    for sample in sampled:
        current_indices.add(sample)
    return current_indices

def experiment3_traffic_rate(real_target, fake_target):
    features = np.array(real_target + fake_target)
    labels = np.array([1] * len(real_target) + [0] * len(fake_target))
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.33,
                                                        random_state=42)
    fittedModel = fitModel(X_train, y_train)
    y_predicted = predictWith(fittedModel, X_test)
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    return y_test, y_predicted

def experiment3_packet_info(real_target, fake_target, numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts):
    features_0 = np.concatenate((np.array(real_target[0]), np.array(fake_target[0])), axis=0)
    features_1 = np.concatenate((np.array(real_target[1]), np.array(fake_target[1])), axis=0)
    features_2 = np.concatenate((np.array(real_target[2]), np.array(fake_target[2])), axis=0)
    features_3 = np.concatenate((np.array(real_target[3]), np.array(fake_target[3])), axis=0)
    features_4 = np.concatenate((np.array(real_target[4]), np.array(fake_target[4])), axis=0)
    features_5 = np.concatenate((np.array(real_target[5]), np.array(fake_target[5])), axis=0)
    features_6 = np.concatenate((np.array(real_target[6]), np.array(fake_target[6])), axis=0)
    labels = np.array([1] * len(real_target[0]) + [0] * len(fake_target[0]))
    X_train_0, X_test_0, X_train_1, X_test_1, X_train_2, X_test_2, X_train_3, X_test_3, X_train_4, X_test_4, X_train_5, X_test_5, X_train_6, X_test_6, y_train, y_test = train_test_split(
        np.array(features_0), np.array(features_1), np.array(features_2), np.array(features_3), np.array(features_4),
        np.array(features_5), np.array(features_6), np.array(labels), test_size=0.33, random_state=42)
    model = packetInfoModel(numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts)
    model.fit([X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6], y_train, batch_size=32,
              epochs=30)
    y_predicted = model.predict([X_test_0, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6], verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    return y_test, y_predicted

def experiment3_sig_freq(real_target, fake_target):
    features = np.concatenate((np.array(real_target), np.array(fake_target)), axis=0)
    labels = np.array([1] * len(real_target) + [0] * len(fake_target))
    X_train, X_test, y_train, y_test = train_test_split(features, np.array(labels), test_size=0.33,
                                                        random_state=42)
    model = signature_frequency_model()
    model.fit(X_train, y_train, batch_size=32, epochs=30)
    y_predicted = model.predict(X_test, verbose=0)[:, 0]
    y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
    return y_test, y_predicted

def evaluate_experiment_3_traffic_rate(real_target, all_fake_target):
    sample_indices = random.sample(range(len(all_fake_target)), len(real_target))
    current_indices = set(sample_indices)
    previous_indices = None
    fake_target = map(lambda x: all_fake_target[x], current_indices)
    previous_y_test = None
    previous_y_predicted = None
    previousAccuracy = 1.0
    for i in range(300):
        y_test, y_predicted = experiment3_traffic_rate(real_target, fake_target)
        acc = accuracy_score(y_test, y_predicted)
        if acc <= previousAccuracy:
            previous_y_test = y_test
            previous_y_predicted = y_predicted
            previous_indices = current_indices
            previousAccuracy = acc
        if i < 100:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 5)
        if i >= 100 and i < 200:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 3)
        if i >= 200:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 1)
        fake_target = map(lambda x: all_fake_target[x], current_indices)
    return {"accuracy": accuracy_score(previous_y_test, previous_y_predicted), "f1": f1_score(previous_y_test, previous_y_predicted)}

def evaluate_experiment_3_packet_info(real_target, all_fake_target, numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts):
    sample_indices = random.sample(range(len(all_fake_target)), len(real_target))
    current_indices = set(sample_indices)
    previous_indices = None
    fake_target = map(lambda x: all_fake_target[x], current_indices)
    previous_y_test = None
    previous_y_predicted = None
    previousAccuracy = 1.0
    for i in range(300):
        y_test, y_predicted = experiment3_packet_info(real_target, fake_target, numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts)
        acc = accuracy_score(y_test, y_predicted)
        if acc <= previousAccuracy:
            previous_y_test = y_test
            previous_y_predicted = y_predicted
            previous_indices = current_indices
            previousAccuracy = acc
        if i < 100:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 5)
        if i >= 100 and i < 200:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 3)
        if i >= 200:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 1)
        fake_target = map(lambda x: all_fake_target[x], current_indices)
    return {"accuracy": accuracy_score(previous_y_test, previous_y_predicted),
            "f1": f1_score(previous_y_test, previous_y_predicted)}

def evaluate_experiment_3_signature_frequency(real_target, all_fake_target):
    sample_indices = random.sample(range(len(all_fake_target)), len(real_target))
    current_indices = set(sample_indices)
    previous_indices = None
    fake_target = map(lambda x: all_fake_target[x], current_indices)
    previous_y_test = None
    previous_y_predicted = None
    previousAccuracy = 1.0
    for i in range(300):
        y_test, y_predicted = experiment3_sig_freq(real_target, fake_target)
        acc = accuracy_score(y_test, y_predicted)
        if acc <= previousAccuracy:
            previous_y_test = y_test
            previous_y_predicted = y_predicted
            previous_indices = current_indices
            previousAccuracy = acc
        if i < 100:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 5)
        if i >= 100 and i < 200:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 3)
        if i >= 200:
            current_indices = swap(previous_indices, range(len(all_fake_target)), 1)
        fake_target = map(lambda x: all_fake_target[x], current_indices)
    return {"accuracy": accuracy_score(previous_y_test, previous_y_predicted), "f1": f1_score(previous_y_test, previous_y_predicted)}

def signature_frequency_model():
    model = Sequential()
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def packetInfoModel(numberOfPacketFrames, numberOfSrcPorts, numberOfDstPorts, numberOfProtocols = len(protocol_list), MAX_LEN=20):
    inp_packet_frame_length = Input(shape=(MAX_LEN,))
    inp_packet_frame_length_continuous = Input(shape=(MAX_LEN, 1))
    inp_direction = Input(shape=(MAX_LEN, 1))
    in_src_ports = Input(shape=(MAX_LEN,))
    in_dst_ports = Input(shape=(MAX_LEN,))
    in_protocols = Input(shape=(MAX_LEN, numberOfProtocols))
    in_durs = Input(shape=(MAX_LEN, 1))
    embedding1 = Embedding(numberOfPacketFrames + 1, 32, input_length=MAX_LEN)(inp_packet_frame_length)
    embedding2 = Embedding(numberOfSrcPorts + 1, 32, input_length=MAX_LEN)(in_src_ports)
    embedding3 = Embedding(numberOfDstPorts + 1, 32, input_length=MAX_LEN)(in_dst_ports)
    merged = Concatenate(axis=2)([embedding1, embedding2, embedding3, inp_packet_frame_length_continuous, inp_direction, in_protocols, in_durs])
    merged = LSTM(100)(merged)
    merged = Dense(100, activation='relu')(merged)
    out1 = Dense(1, activation='sigmoid')(merged)
    model = Model([inp_packet_frame_length, inp_packet_frame_length_continuous, inp_direction, in_src_ports, in_dst_ports, in_protocols, in_durs], out1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

duration_cluster_sizes = [25]
n_embeddings = [500, 2000, 5000]
embedding_dims = [128, 512]

all_results = []

for duration_cluster_size in duration_cluster_sizes:
    for n_embedding in n_embeddings:
        for embedding_dim in embedding_dims:
            packetInfoResults, trafficRateResults, signatureFrequencyResults = evaluate_directory("/home/joseph/vqvae/saved_models", str(duration_cluster_size), str(n_embedding), str(embedding_dim))
            all_results.append((packetInfoResults, duration_cluster_size, n_embedding, embedding_dim))

max_result_per_device = dict()
best_hyperparameters = dict()
for results_for_hyp in all_results:
    for device, results in results_for_hyp[0].items():
        if device not in max_result_per_device:
            max_result_per_device[device] = results[3]['accuracy']
            best_hyperparameters[device] = (results_for_hyp[1], results_for_hyp[2], results_for_hyp[3])
        else:
            current = max_result_per_device[device]
            if results[3]['accuracy'] < current:
                max_result_per_device[device] = results[3]['accuracy']
                best_hyperparameters[device] = (results_for_hyp[1], results_for_hyp[2], results_for_hyp[3])

print(best_hyperparameters)
