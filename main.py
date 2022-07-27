from preprocess import extractAllAaltoFeatures
from reconstruct import reconstruct
from embed import embed, getFrameLengthsAndDirection, signatureExtractionAll
from generate import generate
from evaluate import evaluate
import numpy as np
import pickle
import random
import os
import glob

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

def mapTokens(all_tokens, mapping):
    return [[mapping[y] for y in x] for x in all_tokens]

def mapDuration(all_tokens, mapping, max_duration):
    return [[randomize(mapping[y], max_duration) for y in x] for x in all_tokens]

def extractPacketFrames(packet_frame_lengths):
    return [[abs(y) for y in x] for x in packet_frame_lengths]

def extractDirection(packet_frame_lengths):
    return [[True if y > 0 else False for y in x] for x in packet_frame_lengths]

def mergeAll(all_columns):
    indices = dict()
    for columns in all_columns:
        for i in range(len(columns)):
            column = columns[i]
            indices[i] = len(column)
    results = []
    for i in range(len(indices.keys())):
        result = []
        for j in range(indices[i]):
            row = []
            for columns in all_columns:
                row.append(columns[i][j])
            result.append(row)
        results.append(result)
    return results

def randomize(x, m, d=10.0):
    chosen = random.choice(x)
    return random.uniform(chosen - (chosen/d), min(chosen + (chosen/10.0), m))

def tokensToString(tokens):
    return [str(x) for x in tokens]

def writeTokens(real_tokens, location):
    toWrite = "\n".join([" ".join(tokensToString(tokens)) for tokens in real_tokens])
    with open(location, "w") as t:
        t.write(toWrite)
        t.close()

allAaltoFeatures = extractAllAaltoFeatures("devices")
all_frame_lengths_and_directions = []
for device, flows in allAaltoFeatures.items():
    all_frame_lengths_and_directions += getFrameLengthsAndDirection(flows)
    embedded = embed(flows, min_sig_size, max_sig_size, min_cluster, distance_threshold)
    try:
        os.mkdir("save/" + device)
    except OSError as error:
        print(error)
    pickle.dump(embedded, open("save/" + device + "/embedded.pkl", "wb"))
    writeTokens(embedded[0], "save/" + device + "/real_data.txt")

all_signatures = signatureExtractionAll(all_frame_lengths_and_directions, min_sig_size, max_sig_size, min_cluster, distance_threshold)
pickle.dump(all_signatures, open("save/" + "all_signatures.pkl", "wb"))


# Reconsruct and evaluate


all_signatures = pickle.load(open("save/" + "all_signatures.pkl", "rb"))
all_fake = dict()
all_real = dict()

extended = "save/*/"
paths = glob.glob(extended)
for path in paths:
    embedded = pickle.load(open(path + "/embedded.pkl", "rb"))
    payloads = embedded[13]
    packet_frame_and_direction = mapTokens(embedded[3], embedded[4])
    packet_frames = extractPacketFrames(packet_frame_and_direction)
    directions = extractDirection(packet_frame_and_direction)
    durations = mapDuration(embedded[5], embedded[6], max(np.amax(np.array(list(embedded[6].values())).flatten())))
    protocol_confs = mapTokens(embedded[7], embedded[8])
    src_ports = mapTokens(embedded[9], embedded[10])
    dst_ports = mapTokens(embedded[11], embedded[12])
    merged = mergeAll([payloads, packet_frames, directions, durations, protocol_confs, src_ports, dst_ports])
    reconstructed = reconstruct(embedded[0], embedded[0], embedded[1], embedded[2], embedded[3], embedded[7], embedded[8], embedded[9],
                                embedded[11], embedded[5])
    reconstructedPayloads = reconstructed[0]
    reconstructedPacketFrameAndDirection = mapTokens(reconstructed[1], embedded[4])
    reconstructedPacketFrames = extractPacketFrames(reconstructedPacketFrameAndDirection)
    reconstructedDirections = extractDirection(extractDirection(packet_frame_and_direction))
    reconstructedDurations = mapDuration(reconstructed[2], embedded[6], max(np.amax(np.array(list(embedded[6].values())).flatten())))
    reconstructedProtocolConfs = mapTokens(reconstructed[3], embedded[8])
    reconstructedSrcPorts = mapTokens(reconstructed[4], embedded[10])
    reconstructedDstPorts = mapTokens(reconstructed[5], embedded[12])
    mergedReconstructed = mergeAll([reconstructedPayloads, reconstructedPacketFrames, reconstructedDirections, reconstructedDurations, reconstructedProtocolConfs, reconstructedSrcPorts, reconstructedDstPorts])
    all_real[path] = merged
    all_fake[path] = mergedReconstructed

eval_result = evaluate(all_real, all_fake, all_signatures)
print(eval_result)
pickle.dump(eval_result, open("save/" + "final_result.pkl", "wb"))