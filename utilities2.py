import pyshark
import math
import statistics
from sklearn.cluster import DBSCAN, KMeans
import random
import csv
import numpy as np
import glob
from tqdm import tqdm
from itertools import groupby

def flatten(features):
    r = []
    for sequence in features:
        for row in sequence:
            r.append(row)
    return r

def allLengths(features):
    l = []
    for sequence in features:
        l.append(len(sequence))
    return l

# cluster using dbscan, if any leftover cluster using kmeans
def clusterToToken(features, eps, min_samples, chunkSize = 20, numberOfClusters = 5000):
    flattenFeatures = flatten(features)
    db = DBSCAN(eps, min_samples).fit(flattenFeatures)
    maxClusterNumber = 0
    db_scan_cluster_indices = []
    db_scan_clusters = []
    leftover_features = []
    leftover_clusters = []
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            maxClusterNumber = max(maxClusterNumber, db.labels_[i])
            db_scan_cluster_indices.append(i)
            db_scan_clusters.append(db.labels_[i])
        else:
            leftover_features.append(flattenFeatures[i])
    km = KMeans(n_clusters=(numberOfClusters - maxClusterNumber), random_state=1021).fit(leftover_features)
    for i in range(len(km.labels_)):
        leftover_clusters.append(km.labels_[i])
    db_scan_index = 0
    leftover_index = 0
    tokens = []
    for i in range(len(flattenFeatures)):
        if i == db_scan_cluster_indices[db_scan_index]:
            tokens.append(db_scan_clusters[db_scan_index])
            db_scan_index += 1
        else:
            tokens.append(leftover_clusters[leftover_index])
            leftover_index += 1
    return [tokens[i:i + chunkSize] for i in range(0, len(tokens), chunkSize)]

def labelSignatures(sequences, all_signatures, minSigSize, maxSigSize):
    all_labels = []
    sigFeatureSize = getSignatureFeatureSize(all_signatures, minSigSize, maxSigSize)
    for sequence in sequences:
        labels = len(sequence) * [sigFeatureSize * [0]]
        featureIndex = 0
        for signatureLength in tqdm(range(minSigSize, maxSigSize + 1)):
            signaturesForLength = all_signatures[signatureLength]
            ngramsInSequence = ngrams(signatureLength, sequence)
            for sequenceIndex in range(len(ngramsInSequence)):
                ngram = ngramsInSequence[sequenceIndex]
                for signature in signaturesForLength:
                    if matches(ngram, signature):
                        for ngramIndex in range(signatureLength):
                            idx = sequenceIndex + ngramIndex
                            currentFeats = labels[idx]
                            currentFeats[featureIndex + ngramIndex] = 1
                            labels[idx] = currentFeats
                    featureIndex += signatureLength
        all_labels.append(labels)
    return all_labels

def getRangeFromRanges(ranges):
    mins = []
    maxs = []
    for range in ranges:
        mins.append(range[0])
        maxs.append(range[1])
    return [max(mins), min(maxs)]

def getSignatureIndexes(all_signatures, minSigSize, maxSigSize):
    ranges = []
    for signatureLength in tqdm(range(minSigSize, maxSigSize + 1)):
        signaturesForLength = all_signatures[signatureLength]
        for signature in signaturesForLength:
            for range in signature:
                ranges.append(range)
    return ranges

def getSignatureFeatureSize(all_signatures, minSigSize, maxSigSize):
    total = 0
    for signatureLength in tqdm(range(minSigSize, maxSigSize + 1)):
        total += len(all_signatures[signatureLength]) * signatureLength
    return total

def matches(ngram, signature):
    if len(ngram) != len(signature):
        return False
    for i in range(len(ngram)):
        ngramElement = ngram[i]
        signatureElement = signature[i]
        sigMin = signatureElement[0]
        sigMax = signatureElement[1]
        if ngramElement < sigMin or ngramElement > sigMax:
            return False
    return True

def signatureExtractionAll(sequences, minSigSize, maxSigSize, distance_threshold, cluster_threshold):
    all_signatures = dict()
    print("extracting signatures")
    for i in tqdm(range(minSigSize, maxSigSize + 1)):
        allngrams = []
        for sequence in sequences:
            ngramVector = ngrams(i, sequence)
            for ngram in ngramVector:
                allngrams.append(ngram)
        cluster = dbclustermin(allngrams, distance_threshold, cluster_threshold)
        signatures = extractSignatures(cluster, i)
        all_signatures[i] = signatures
    return all_signatures

def extractSignatures(clusters, n):
    signatures = []
    for cluster in clusters:
        signature = []
        for i in range(n):
            column = []
            for seq in cluster:
                column.append(seq[i])
            signature.append((min(column), max(column)))
        signatures.append(signature)
    return signatures

def most_common(lst):
    return max(set(lst), key=lst.count)

def ngrams(n, sequence):
    output = []
    for i in range(len(sequence) - n + 1):
        output.append(sequence[i:i + n])
    return output

def dbclustermin(x, eps, min_samples):
    db = DBSCAN(eps, min_samples).fit(x)
    clusters = dict()
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())

def normalize_durations(sequences):
    max_d = 0.0
    num_seqs = []
    final_num_seqs = []
    for sequence in sequences:
        num_seq = [float(x) for x in sequence]
        max_d = max(max(num_seq), max_d)
        num_seqs.append(num_seq)
    for num_seq in num_seqs:
        final_num_seq = [x/max_d for x in num_seq]
        final_num_seqs.append(final_num_seq)
    return final_num_seqs, max_d

def durationcluster(x, n_clusters=20):
    x = [i for i in x if i != 0]
    newX = np.array(x)
    newX = np.log(newX)
    newX = np.expand_dims(newX, axis=1)
    clusters = dict()
    db = KMeans(n_clusters=n_clusters, random_state=1021).fit(newX)
    for i in range(len(db.labels_)):
        clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())

def signatureToString(signature):
    signature_ints = []
    for tuple in signature:
        signature_ints.append(tuple[0])
        signature_ints.append(tuple[1])
    return ', '.join(str(x) for x in signature_ints)

def stringToSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    int_arr = [int(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(int_arr), 2):
        sig.append((int_arr[i], int_arr[i + 1]))
    return sig

def toDurationRanges(clusters):
    rangesToTokens = dict()
    tokensToRanges = dict()
    tokensToMean = dict()
    tokensTostd = dict()
    zeroRange = signatureToString([(0, 0)])
    rangesToTokens[zeroRange] = 0
    tokensToRanges[0] = zeroRange
    tokensToMean[0] = 0
    tokensTostd[0] = 0
    for i in range(len(clusters)):
        cluster = clusters[i]
        clusMin = min(cluster)
        clusMax = max(cluster)
        mean = statistics.mean(cluster)
        if len(cluster) > 1:
            std = statistics.stdev(cluster)
        else:
            std = 0
        rangeString = signatureToString([(clusMin, clusMax)])
        rangesToTokens[rangeString] = i + 1
        tokensToRanges[i + 1] = rangeString
        tokensToMean[i + 1] = mean
        tokensTostd[i + 1] = std
    return rangesToTokens, tokensToRanges