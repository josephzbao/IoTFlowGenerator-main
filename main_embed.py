from embed import embed
from preprocess import extractAllAaltoFeatures
import pickle
import csv
import os

devices_path = "devices"

def write_tokens(all_tokens, location):
    with open(location, mode='w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerows(all_tokens)

all_features_all_devices = extractAllAaltoFeatures(devices_path)
for device, features in all_features_all_devices.items():
    if device.endswith("pkl"):
        continue
    device_path = devices_path + "/" + device
    print("device")
    print(device)
    duration_cluster_sizes = [25, 50, 100]
    for duration_cluster_size in duration_cluster_sizes:
        embeddingResult = embed(features, 2, 5, 4, 5, duration_cluster_size)
        if embeddingResult is None:
            continue
        write_tokens(embeddingResult[0], device_path + "/real_data.txt")
        os.makedirs(device_path + "/" + str(duration_cluster_size), exist_ok=True)
        with open(device_path + "/" + str(duration_cluster_size) + "/real_features.pkl", mode='wb') as pklfile:
            pickle.dump(embeddingResult, pklfile)
            pklfile.close()
        with open(device_path + "/all_signatures.pkl", mode='wb') as pklfile:
            pickle.dump(embeddingResult[14], pklfile)
            pklfile.close()
        with open(device_path + "/" + str(duration_cluster_size) + "/max_duration.pkl", mode='wb') as pklfile:
            pickle.dump(embeddingResult[15], pklfile)
            pklfile.close()