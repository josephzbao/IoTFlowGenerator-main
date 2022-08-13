# Creates a discrete mapping
import statistics

from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import random
from sklearn_extra.cluster import KMedoids
from data_prep import normalize_packet_sizes, get_max_packet_size
from pcaputilities import map_all_signatures, all_greedy_activity_conversion, convert_sig_sequences_to_ranges, extract_dictionaries_from_activities
from preprocess import extractAllAaltoFeatures

# distance metric used by dbscan
distance_threshold = 5.0
# total ngrams divided by cluster threshold is equal to the min_samples needed to form a cluster in dbscan
min_cluster = 4
min_sig_size = 2
max_sig_size = 5

def is_valid_seqs(sequences):
    for seq in sequences:
        if len(seq) >= 20:
            return True
    return False

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
                if isinstance(columns[i][j], list):
                    row += columns[i][j]
                else:
                    row.append(columns[i][j])
            result.append(row)
        results.append(result)
    return results

def normalize_any(all_features):
    singular = []
    for features in all_features:
        singular += features
    return singular

def getFrameLengthsAndDirection(trafficFlowFeatures):
    trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, trafficFlowFeatures))
    frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
    return frameLengthsAndDirection

def embed(trafficFlowFeatures, minSigSize, maxSigSize, distance_threshold, cluster_threshold, duration_cluter_size):
    trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, trafficFlowFeatures))
    frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
    max_packet_size = get_max_packet_size(extractPacketFrameLengthsWithDirection(trafficFlowFeatures))
    normalized_p = normalize_packet_sizes(frameLengthsAndDirection, max_packet_size)
    all_signatures = signatureExtractionAll(normalized_p, minSigSize, maxSigSize, distance_threshold, cluster_threshold)
    range_mapping = map_all_signatures(all_signatures)
    range_mapping_results = all_greedy_activity_conversion(normalized_p, all_signatures)
    all_ranges_results = convert_sig_sequences_to_ranges(range_mapping_results, range_mapping)
    rangesToPacketTokens, packetTokensToRanges = extract_dictionaries_from_activities(all_ranges_results)
    all_rangeSequences = []
    for sequence in all_ranges_results:
        rans = []
        for ran in sequence:
            rans.append(rangesToPacketTokens[ran])
        all_rangeSequences.append(rans)
    if len(np.array(frameLengthsAndDirection).flatten()) == 0:
        signatureRows = frameLengthsAndDirection
        sigFeatureSize = 0
    else:
        signatureRows, sigFeatureSize, all_signatures = signatureFeatures(frameLengthsAndDirection, minSigSize, maxSigSize, distance_threshold, cluster_threshold)
    all_full_signature_labels = []
    all_packet_frame_range_tokens = []
    all_protocol_conf_tokens = []
    all_range_tokens = []
    signaturesStringMappedToToken = dict()
    signatureStringMappedToSignatureRow = dict()
    nullSignatureString = intArrayToString(sigFeatureSize * [0])
    signatureStringMappedToSignatureRow[nullSignatureString] = sigFeatureSize * [0]
    signaturesStringMappedToToken[nullSignatureString] = 0
    protocolStringMappedToToken = dict()
    protocolStringMappedToProtocolConf = dict()
    srcPortMappedToToken = dict()
    dstPortMappedToToken = dict()
    range_token = 1
    protocol_token = 0
    src_port_token = 0
    dst_port_token = 0
    all_protocol_confs = []
    all_src_port_tokens = []
    all_dst_port_tokens = []
    for i in range(len(trafficFlowFeatures)):
        full_signature_labels = []
        packet_frame_tokens = []
        packet_range_tokens = []
        protocol_confs = []
        protocol_conf_tokens = []
        src_port_tokens = []
        dst_port_tokens = []
        trafficFlow = trafficFlowFeatures[i]
        signatureRow = signatureRows[i]
        packet_range_row = all_rangeSequences[i]
        signatureIndex = 0
        packet_range_index = 0
        for packet in trafficFlow:
            # signatures
            if packet[0] is True:
                signatureLabel = signatureRow[signatureIndex]
                full_signature_labels.append(signatureLabel)
                packet_range_tokens.append(packet_range_row[packet_range_index])
                signatureIndex += 1
                packet_range_index += 1
                signatureRowString = intArrayToString(signatureLabel)
                if signatureRowString not in signaturesStringMappedToToken.keys():
                    signaturesStringMappedToToken[signatureRowString] = range_token
                    signatureStringMappedToSignatureRow[signatureRowString] = signatureLabel
                    packet_frame_tokens.append(range_token)
                    range_token += 1
                else:
                    packet_frame_tokens.append(signaturesStringMappedToToken[signatureRowString])
            else:
                full_signature_labels.append(sigFeatureSize * [0])
                packet_frame_tokens.append(0)
                packet_range_tokens.append(-1)
            # protocols
            protocol_conf = packet[4]
            protocol_conf_string = intArrayToString(protocol_conf)
            protocol_confs.append(protocol_conf)
            if protocol_conf_string not in protocolStringMappedToToken.keys():
                protocolStringMappedToToken[protocol_conf_string] = protocol_token
                protocolStringMappedToProtocolConf[protocol_conf_string] = protocol_conf
                protocol_conf_tokens.append(protocol_token)
                protocol_token += 1
            else:
                protocol_conf_tokens.append(protocolStringMappedToToken[protocol_conf_string])
            # src port
            src_port = packet[5]
            if src_port not in srcPortMappedToToken.keys():
                srcPortMappedToToken[src_port] = src_port_token
                src_port_tokens.append(src_port_token)
                src_port_token += 1
            else:
                src_port_tokens.append(srcPortMappedToToken[src_port])
            # dst port
            dst_port = packet[6]
            if dst_port not in dstPortMappedToToken.keys():
                dstPortMappedToToken[dst_port] = dst_port_token
                dst_port_tokens.append(dst_port_token)
                dst_port_token += 1
            else:
                dst_port_tokens.append(dstPortMappedToToken[dst_port])
        all_full_signature_labels.append(full_signature_labels)
        all_packet_frame_range_tokens.append(packet_frame_tokens)
        all_protocol_conf_tokens.append(protocol_conf_tokens)
        all_protocol_confs.append(protocol_confs)
        all_src_port_tokens.append(src_port_tokens)
        all_dst_port_tokens.append(dst_port_tokens)
        all_range_tokens.append(packet_range_tokens)
    tokenToSignatureRow = {v: signatureStringMappedToSignatureRow[k] for k, v in signaturesStringMappedToToken.items()}
    tokensToProtocolConfs = {v: protocolStringMappedToProtocolConf[k] for k, v in protocolStringMappedToToken.items()}
    tokensToSrcPort = {v: k for k, v in srcPortMappedToToken.items()}
    tokensToDstPort = {v: k for k, v in dstPortMappedToToken.items()}
    all_durations = extractDuration(trafficFlowFeatures)
    all_payloads = extractPayload(trafficFlowFeatures)
    max_duration = max(flatten(all_durations))
    all_normalized_durations = normalize_durations(all_durations, max_duration)
    duration_clusters = durationcluster(flatten(all_normalized_durations), duration_cluter_size)
    rangesToTokens, tokensToRanges = toDurationRanges(duration_clusters)
    all_duration_tokens, all_tokens_to_durations = convertalldurationstoint(all_normalized_durations, rangesToTokens)
    frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeatures)
    packet_frame_tokens, tokens_to_packet_frame_lengths = encodePacketFrames(frameLengthsAndDirection)
    # clusterTokens = clusterDiffMatrix(normalize_any(all_full_signature_labels), normalize_any(all_payloads), frameLengthsAndDirection, normalize_any(all_normalized_durations), normalize_any(all_protocol_confs), normalize_any(all_src_port_tokens), normalize_any(all_dst_port_tokens))
    # print("clusterTokens")
    # print(clusterTokens)
    # clusterFeatures = mergeAll([all_full_signature_labels, NormalizeData(frameLengthsAndDirection), all_normalized_durations, all_protocol_confs, OneHotEncode(all_src_port_tokens, src_port_token), OneHotEncode(all_dst_port_tokens, dst_port_token)])
    # clusterTokens = clusterToToken(clusterFeatures)
    if not is_valid_seqs(all_dst_port_tokens):
        return None
    selectionResult = randomlySelect(
        [all_dst_port_tokens, all_dst_port_tokens, all_dst_port_tokens, all_dst_port_tokens,
         all_src_port_tokens, all_dst_port_tokens, np.array(all_packet_frame_range_tokens),
         np.array(packet_frame_tokens), np.array(all_duration_tokens), np.array(all_protocol_conf_tokens),
         np.array(all_payloads), np.array(all_range_tokens)])
    # all_full_signature_labels = selectionResult[0]
    # frameLengthsAndDirection = selectionResult[1]
    # all_normalized_durations = selectionResult[2]
    # all_protocol_confs = selectionResult[3]
    all_src_port_tokens = selectionResult[4]
    all_dst_port_tokens = selectionResult[5]
    all_packet_frame_range_tokens = selectionResult[6]
    packet_frame_tokens = selectionResult[7]
    all_duration_tokens = selectionResult[8]
    all_protocol_conf_tokens = selectionResult[9]
    all_payloads = selectionResult[10]
    all_range_tokens = selectionResult[11]
    clusterTokens = []
    return clusterTokens, all_packet_frame_range_tokens, tokenToSignatureRow, packet_frame_tokens, tokens_to_packet_frame_lengths, all_duration_tokens, all_tokens_to_durations, all_protocol_conf_tokens, tokensToProtocolConfs, all_src_port_tokens, tokensToSrcPort, all_dst_port_tokens, tokensToDstPort, all_payloads, all_signatures, max_duration, all_range_tokens, packetTokensToRanges, max_packet_size


def embed2(trafficFlowFeatures, minSigSize, maxSigSize, distance_threshold, cluster_threshold):
    trafficFlowFeaturesWithNoisyFiltered = list(map(filterNoisy, trafficFlowFeatures))
    frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeaturesWithNoisyFiltered)
    normalized_p, max_packet_size = normalize_packet_sizes(extractPacketFrameLengthsWithDirection(trafficFlowFeatures))
    all_signatures = signatureExtractionAll(normalized_p, minSigSize, maxSigSize, distance_threshold, cluster_threshold)
    range_mapping = map_all_signatures(all_signatures)
    range_mapping_results = all_greedy_activity_conversion(normalized_p, all_signatures)
    all_ranges_results = convert_sig_sequences_to_ranges(range_mapping_results, range_mapping)
    rangesToTokens, tokensToRanges = extract_dictionaries_from_activities(all_ranges_results)
    all_rangeSequences = []
    for sequence in all_ranges_results:
        rans = []
        for ran in sequence:
            rans.append(rangesToTokens[ran])
        all_rangeSequences.append(rans)
    if len(np.array(frameLengthsAndDirection).flatten()) == 0:
        signatureRows = frameLengthsAndDirection
        sigFeatureSize = 0
    else:
        signatureRows, sigFeatureSize, all_signatures = signatureFeatures(frameLengthsAndDirection, minSigSize, maxSigSize, distance_threshold, cluster_threshold)
    all_full_signature_labels = []
    all_packet_frame_range_tokens = []
    all_protocol_conf_tokens = []
    all_range_tokens = []
    signaturesStringMappedToToken = dict()
    signatureStringMappedToSignatureRow = dict()
    nullSignatureString = intArrayToString(sigFeatureSize * [0])
    signatureStringMappedToSignatureRow[nullSignatureString] = sigFeatureSize * [0]
    signaturesStringMappedToToken[nullSignatureString] = 0
    protocolStringMappedToToken = dict()
    protocolStringMappedToProtocolConf = dict()
    srcPortMappedToToken = dict()
    dstPortMappedToToken = dict()
    range_token = 1
    protocol_token = 0
    src_port_token = 0
    dst_port_token = 0
    all_protocol_confs = []
    all_src_port_tokens = []
    all_dst_port_tokens = []
    for i in range(len(trafficFlowFeatures)):
        full_signature_labels = []
        packet_frame_tokens = []
        packet_range_tokens = []
        protocol_confs = []
        protocol_conf_tokens = []
        src_port_tokens = []
        dst_port_tokens = []
        trafficFlow = trafficFlowFeatures[i]
        signatureRow = signatureRows[i]
        packet_range_row = all_rangeSequences[i]
        signatureIndex = 0
        packet_range_index = 0
        for packet in trafficFlow:
            # signatures
            if packet[0] is True:
                signatureLabel = signatureRow[signatureIndex]
                full_signature_labels.append(signatureLabel)
                packet_range_tokens.append(packet_range_row[packet_range_index])
                signatureIndex += 1
                packet_range_index += 1
                signatureRowString = intArrayToString(signatureLabel)
                if signatureRowString not in signaturesStringMappedToToken.keys():
                    signaturesStringMappedToToken[signatureRowString] = range_token
                    signatureStringMappedToSignatureRow[signatureRowString] = signatureLabel
                    packet_frame_tokens.append(range_token)
                    range_token += 1
                else:
                    packet_frame_tokens.append(signaturesStringMappedToToken[signatureRowString])
            else:
                full_signature_labels.append(sigFeatureSize * [0])
                packet_frame_tokens.append(0)
                packet_range_tokens.append(-1)
            # protocols
            protocol_conf = packet[4]
            protocol_conf_string = intArrayToString(protocol_conf)
            protocol_confs.append(protocol_conf)
            if protocol_conf_string not in protocolStringMappedToToken.keys():
                protocolStringMappedToToken[protocol_conf_string] = protocol_token
                protocolStringMappedToProtocolConf[protocol_conf_string] = protocol_conf
                protocol_conf_tokens.append(protocol_token)
                protocol_token += 1
            else:
                protocol_conf_tokens.append(protocolStringMappedToToken[protocol_conf_string])
            # src port
            src_port = packet[5]
            if src_port not in srcPortMappedToToken.keys():
                srcPortMappedToToken[src_port] = src_port_token
                src_port_tokens.append(src_port_token)
                src_port_token += 1
            else:
                src_port_tokens.append(srcPortMappedToToken[src_port])
            # dst port
            dst_port = packet[6]
            if dst_port not in dstPortMappedToToken.keys():
                dstPortMappedToToken[dst_port] = dst_port_token
                dst_port_tokens.append(dst_port_token)
                dst_port_token += 1
            else:
                dst_port_tokens.append(dstPortMappedToToken[dst_port])
        all_full_signature_labels.append(full_signature_labels)
        all_packet_frame_range_tokens.append(packet_frame_tokens)
        all_protocol_conf_tokens.append(protocol_conf_tokens)
        all_protocol_confs.append(protocol_confs)
        all_src_port_tokens.append(src_port_tokens)
        all_dst_port_tokens.append(dst_port_tokens)
        all_range_tokens.append(packet_range_tokens)
    tokenToSignatureRow = {v: signatureStringMappedToSignatureRow[k] for k, v in signaturesStringMappedToToken.items()}
    tokensToProtocolConfs = {v: protocolStringMappedToProtocolConf[k] for k, v in protocolStringMappedToToken.items()}
    tokensToSrcPort = {v: k for k, v in srcPortMappedToToken.items()}
    tokensToDstPort = {v: k for k, v in dstPortMappedToToken.items()}
    all_durations = extractDuration(trafficFlowFeatures)
    all_payloads = extractPayload(trafficFlowFeatures)
    max_duration = max(flatten(all_durations))
    all_normalized_durations = normalize_durations(all_durations, max_duration)
    duration_clusters = durationcluster(flatten(all_normalized_durations))
    rangesToTokens, tokensToRanges = toDurationRanges(duration_clusters)
    all_duration_tokens, all_tokens_to_durations = convertalldurationstoint(all_normalized_durations, rangesToTokens)
    frameLengthsAndDirection = extractPacketFrameLengthsWithDirection(trafficFlowFeatures)
    packet_frame_tokens, tokens_to_packet_frame_lengths = encodePacketFrames(frameLengthsAndDirection)
    clusterTokens = clusterDiffMatrix(normalize_any(all_full_signature_labels), normalize_any(all_payloads), frameLengthsAndDirection, normalize_any(all_normalized_durations), normalize_any(all_protocol_confs), normalize_any(all_src_port_tokens), normalize_any(all_dst_port_tokens))
    print("clusterTokens")
    print(clusterTokens)
    # clusterFeatures = mergeAll([all_full_signature_labels, NormalizeData(frameLengthsAndDirection), all_normalized_durations, all_protocol_confs, OneHotEncode(all_src_port_tokens, src_port_token), OneHotEncode(all_dst_port_tokens, dst_port_token)])
    # clusterTokens = clusterToToken(clusterFeatures)
    selectionResult = randomlySelect(
        [all_full_signature_labels, frameLengthsAndDirection, all_normalized_durations, all_protocol_confs,
         all_src_port_tokens, all_dst_port_tokens, np.array(all_packet_frame_range_tokens),
         np.array(packet_frame_tokens), np.array(all_duration_tokens), np.array(all_protocol_conf_tokens),
         np.array(all_payloads), clusterTokens])
    # all_full_signature_labels = selectionResult[0]
    # frameLengthsAndDirection = selectionResult[1]
    # all_normalized_durations = selectionResult[2]
    # all_protocol_confs = selectionResult[3]
    all_src_port_tokens = selectionResult[4]
    all_dst_port_tokens = selectionResult[5]
    all_packet_frame_range_tokens = selectionResult[6]
    packet_frame_tokens = selectionResult[7]
    all_duration_tokens = selectionResult[8]
    all_protocol_conf_tokens = selectionResult[9]
    all_payloads = selectionResult[10]
    clusterTokens = selectionResult[11]
    return clusterTokens, all_packet_frame_range_tokens, tokenToSignatureRow, packet_frame_tokens, tokens_to_packet_frame_lengths, all_duration_tokens, all_tokens_to_durations, all_protocol_conf_tokens, tokensToProtocolConfs, all_src_port_tokens, tokensToSrcPort, all_dst_port_tokens, tokensToDstPort, all_payloads, all_signatures, max_duration, all_range_tokens, tokensToRanges, max_packet_size


def computeDistanceMatrix(all_full_signature_labels, all_payloads, frameLengthsAndDirection, all_normalized_durations, all_protocol_confs, all_src_port_tokens, all_dst_port_tokens):
    maxFrame = abs(max(frameLengthsAndDirection, key=abs))
    diffMatrix = []
    for i in range(len(all_full_signature_labels)):
        diffRow = []
        for j in range(len(all_full_signature_labels)):
            sigs1 = all_full_signature_labels[i]
            sigs2 = all_full_signature_labels[j]
            signatureDifference = sum([abs(a_i - b_i) for a_i, b_i in zip(sigs1, sigs2)])
            signatureSum = sum(sigs1) + sum(sigs2)
            signatureDifference = 0 if signatureDifference == 0 or signatureSum == 0 else float(signatureDifference)/float(signatureSum)
            payloads1 = all_payloads[i]
            payloads2 = all_payloads[j]
            payloadDifference = 0 if payloads1 == payloads2 else 1
            frameLengthAndDirection1 = frameLengthsAndDirection[i]
            frameLengthAndDirection2 = frameLengthsAndDirection[j]
            if frameLengthAndDirection1 * frameLengthAndDirection2 < 0:
                frameDifference = 1
            else:
                frameDifference = float(abs(frameLengthAndDirection1 - frameLengthAndDirection2))/float(maxFrame)
            dur1 = all_normalized_durations[i]
            dur2 = all_normalized_durations[j]
            durDifference = abs(dur1 - dur2)
            protocol_conf1 = all_protocol_confs[i]
            protocol_conf2 = all_protocol_confs[j]
            protocol_conf_difference = sum([abs(a_i - b_i) for a_i, b_i in zip(protocol_conf1, protocol_conf2)])
            protocol_conf_sum = sum(protocol_conf1) + sum(protocol_conf2)
            protocol_conf_difference = 0 if protocol_conf_difference == 0 or protocol_conf_sum == 0 else float(protocol_conf_difference)/float(protocol_conf_sum)
            src_port1 = all_src_port_tokens[i]
            src_port2 = all_src_port_tokens[j]
            dst_port1 = all_dst_port_tokens[i]
            dst_port2 = all_dst_port_tokens[j]
            port_difference = 0 if src_port1 == src_port2 or dst_port1 == dst_port2 else 1
            if port_difference == 1 or payloadDifference == 1 or protocol_conf_difference == 1:
                difference = 1
            else:
                signatureWeight = 10.0
                frameWeight = 10.0
                durWeight = 10.0
                totalWeight = signatureWeight + frameWeight + durWeight
                difference = ((signatureWeight/totalWeight) * signatureDifference) + ((frameWeight/totalWeight) * frameDifference) + ((durWeight/totalWeight) * durDifference)
            diffRow.append(difference)
        diffMatrix.append(diffRow)
    return diffMatrix

def randomlySelect(all_sequences, length=20, repeat=500):
    rows = dict()
    columns = dict()
    token = 0
    sequences = all_sequences[0]
    for i in range(len(sequences)):
        sequence = sequences[i]
        for j in range(len(sequence) - (length - 1)):
            rows[token] = i
            columns[token] = j
            token += 1
    randomSelections = []
    for _ in range(repeat):
        index = random.randint(0, token-1)
        randomSelections.append(index)
    all_selections = []
    for seqs in all_sequences:
        selections = []
        for selection in randomSelections:
            row = rows[selection]
            column = columns[selection]
            selections.append(seqs[row][column:column+length])
        all_selections.append(selections)
    return all_selections

def normalize_durations(all_durations, max_duration):
    all_normalized_durations = []
    for durations in all_durations:
        normalized_durations = []
        for duration in durations:
            normalized_durations.append(duration / max_duration)
        all_normalized_durations.append(normalized_durations)
    return all_normalized_durations

def keepIndicesFromSquare(distanceMatrix, indices):
    indexSet = set(indices)
    newDistanceMatrix = []
    for i in range(len(distanceMatrix)):
        if i in indexSet:
            row = []
            for j in range(len(distanceMatrix)):
                if j in indexSet:
                    row.append(distanceMatrix[i][j])
            newDistanceMatrix.append(row)
    return newDistanceMatrix

def clusterDiffMatrix(all_full_signature_labels, all_payloads, frameLengthsAndDirection, all_normalized_durations, all_protocol_confs, all_src_port_tokens, all_dst_port_tokens, min_cluster_size=5, numberOfClusters = 5000):
    print("frame lengths and direction")
    print(len(normalize_any(frameLengthsAndDirection)))
    distanceMatrix = computeDistanceMatrix(all_full_signature_labels, all_payloads, normalize_any(frameLengthsAndDirection), all_normalized_durations, all_protocol_confs, all_src_port_tokens, all_dst_port_tokens)
    print("distance matrix")
    print(len(distanceMatrix))
    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_method="eom", metric="precomputed").fit(np.array(distanceMatrix))
    db_scan_cluster_indices = []
    db_scan_clusters = []
    no_cluster_indices = []
    maxClusterNumber = 0
    leftover_clusters = []
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            maxClusterNumber = max(maxClusterNumber, db.labels_[i])
            db_scan_cluster_indices.append(i)
            db_scan_clusters.append(db.labels_[i])
        else:
            no_cluster_indices.append(i)
    distanceMatrix = keepIndicesFromSquare(distanceMatrix, no_cluster_indices)
    if len(no_cluster_indices) > 2:
        km = KMedoids(metric="precomputed", n_clusters=min(200, (numberOfClusters - maxClusterNumber), len(no_cluster_indices)), random_state=1021).fit(distanceMatrix)
        for i in range(len(km.labels_)):
            leftover_clusters.append(maxClusterNumber + 1 + km.labels_[i])
    else:
        leftover_clusters = (len(no_cluster_indices) * [maxClusterNumber + 1])
    print("distance matrix")
    print(len(distanceMatrix))
    print("leftover")
    print(len(leftover_clusters))
    print("dbscan clusters")
    print(len(db_scan_clusters))
    db_scan_index = 0
    leftover_index = 0
    tokens = []
    for i in range(len(all_payloads)):
        if db_scan_index < len(db_scan_cluster_indices) and i == db_scan_cluster_indices[db_scan_index]:
            tokens.append(db_scan_clusters[db_scan_index])
            db_scan_index += 1
        else:
            tokens.append(leftover_clusters[leftover_index])
            leftover_index += 1
    idx = 0
    all_tokens = []
    for row in frameLengthsAndDirection:
        all_tokens.append(tokens[idx:idx + len(row)])
        idx += len(row)
    return all_tokens

# cluster using hdbscan/dbscan, if any leftover cluster using kmeans
def clusterToToken(features, min_cluster_size=5, numberOfClusters = 5000):
    flattenFeatures = flatten(features)
    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_method="eom").fit(flattenFeatures)
    # db = DBSCAN(0.1, 5).fit(flattenFeatures)
    print(db.labels_)
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
    if len(leftover_features) > 2:
        km = KMeans(n_clusters=min(1000, (numberOfClusters - maxClusterNumber), len(leftover_features)), random_state=1021).fit(leftover_features)
        for i in range(len(km.labels_)):
            leftover_clusters.append(maxClusterNumber + 1 + km.labels_[i])
    else:
        leftover_clusters = (len(leftover_features) * [maxClusterNumber + 1])
    db_scan_index = 0
    leftover_index = 0
    tokens = []
    for i in range(len(flattenFeatures)):
        if db_scan_index < len(db_scan_cluster_indices) and i == db_scan_cluster_indices[db_scan_index]:
            tokens.append(db_scan_clusters[db_scan_index])
            db_scan_index += 1
        else:
            tokens.append(leftover_clusters[leftover_index])
            leftover_index += 1
    idx = 0
    all_tokens = []
    for row in features:
        all_tokens.append(tokens[idx:idx+len(row)])
        idx += len(row)
    return all_tokens

def flatten(features):
    r = []
    for sequence in features:
        for row in sequence:
            r.append(row)
    return r

def OneHotEncode(all_tokens, tokenMax):
    all_results = []
    for tokens in all_tokens:
        results = []
        for token in tokens:
            onehot = tokenMax * [0]
            onehot[token] = 1
            results.append(onehot)
        all_results.append(results)
    return all_results

def getMax(data):
    return max(map(max, data))

def getMin(data):
    return min(map(min, data))

def NormalizeData(data):
    minGot = getMin(data)
    maxGot = getMax(data)
    if minGot == maxGot:
        return [[y / minGot for y in x] for x in data]
    return [[((y - minGot)/(maxGot - minGot)) for y in x] for x in data]

def encodePacketFrames(frameLengthAndDirection):
    frameLengthAndDirectionToToken = dict()
    token = 0
    for element in set(list(flatten(frameLengthAndDirection))):
        if element not in frameLengthAndDirectionToToken.keys():
            frameLengthAndDirectionToToken[element] = token
            token += 1
    all_tokens = []
    for row in frameLengthAndDirection:
        tokens = []
        for element in row:
            tokens.append(frameLengthAndDirectionToToken[element])
        all_tokens.append(tokens)
    return all_tokens, {v: k for k, v in frameLengthAndDirectionToToken.items()}

def intArrayToString(intArray):
    return ''.join([str(x) for x in intArray])

def extractPayload(trafficFlowFeatures):
    return list(map(keepPayload, trafficFlowFeatures))

def keepPayload(trafficFlow):
    return list(map(lambda x: x[0], trafficFlow))

def filterNoisy(trafficFlow):
    return list(filter(lambda x: x[0] is True, trafficFlow))

def extractPacketFrameLengthsWithDirection(trafficFlowFeatures):
    return list(map(keepPacketFrameLengthsWithDirection, trafficFlowFeatures))

def keepPacketFrameLengthsWithDirection(trafficFlow):
    return list(map(lambda x: (1 if x[2] == True else -1) * x[1], trafficFlow))

def extractDuration(trafficFlowFeatures):
    return list(map(keepDuration, trafficFlowFeatures))

def keepDuration(trafficFlow):
    return list(map(lambda x: x[3], trafficFlow))

def signatureFeatures(frameLengthDirectionFlows, minSigSize, maxSigSize, distance_threshold, cluster_threshold):
    all_signatures = signatureExtractionAll(frameLengthDirectionFlows, minSigSize, maxSigSize, distance_threshold, cluster_threshold)
    return labelSignatures(frameLengthDirectionFlows, all_signatures, minSigSize, maxSigSize)

def labelSignatures(sequences, all_signatures, minSigSize, maxSigSize):
    all_labels = []
    sigFeatureSize = getSignatureFeatureSize(all_signatures, minSigSize, maxSigSize)
    for sequence in sequences:
        startingFeatureIndex = 0
        labels = [[0 for i in range(sigFeatureSize)] for j in range(len(sequence))]
        for signatureLength in tqdm(range(minSigSize, maxSigSize + 1)):
            if signatureLength in all_signatures:
                signaturesForLength = all_signatures[signatureLength]
            else:
                signaturesForLength = []
            ngramsInSequence = ngrams(signatureLength, sequence)
            for sequenceIndex in range(len(ngramsInSequence)):
                featureIndex = 0
                ngram = ngramsInSequence[sequenceIndex]
                for signature in signaturesForLength:
                    if matches(ngram, signature):
                        for ngramIndex in range(signatureLength):
                            idx = sequenceIndex + ngramIndex
                            labels[idx][startingFeatureIndex + featureIndex + ngramIndex] = 1
                    featureIndex = featureIndex + signatureLength
            startingFeatureIndex = startingFeatureIndex + len(signaturesForLength) * signatureLength
        all_labels.append(labels)
    return all_labels, sigFeatureSize, all_signatures

def getSignatureFeatureSize(all_signatures, minSigSize, maxSigSize):
    total = 0
    for signatureLength in tqdm(range(minSigSize, maxSigSize + 1)):
        if signatureLength in all_signatures:
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
        if len(allngrams) > 0:
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

def durationcluster(x, n_clusters=50):
    x = [i for i in x if i != 0]
    newX = np.array(x)
    newX = np.log(newX)
    newX = np.expand_dims(newX, axis=1)
    clusters = dict()
    db = KMeans(n_clusters=n_clusters, random_state=1021).fit(newX)
    # db = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_method="eom").fit(newX)
    print(db.labels_)
    for i in range(len(db.labels_)):
        clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    print(clusters.values())
    print(len(clusters))
    return list(clusters.values())

def dbclustermin(x, eps, min_samples):
    db = DBSCAN(eps, min_samples).fit(x)
    clusters = dict()
    for i in range(len(db.labels_)):
        if db.labels_[i] != -1:
            clusters[db.labels_[i]] = clusters.get(db.labels_[i], []) + [x[i]]
    return list(clusters.values())

def ngrams(n, sequence):
    output = []
    for i in range(len(sequence) - n + 1):
        output.append(sequence[i:i + n])
    return output

def signatureToString(signature):
    signature_ints = []
    for tuple in signature:
        signature_ints.append(tuple[0])
        signature_ints.append(tuple[1])
    return ', '.join(str(x) for x in signature_ints)

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

def convertalldurationstoint(all_durations, rangesToTokens):
    all_tokens = []
    all_tokens_to_durations = dict()
    sortedRanges = sortRanges(rangesToTokens)
    for durations in all_durations:
        tokens, tokensToDurations = convertdurationsToInt(durations, sortedRanges)
        for key, value in tokensToDurations.items():
            all_tokens_to_durations[key] = all_tokens_to_durations.get(key, []) + value
        all_tokens.append(tokens)
    return all_tokens, all_tokens_to_durations

def convertdurationsToInt(durations, sortedRanges):
    tokens = []
    tokensToDurations = dict()
    for duration in durations:
        for key, value in sortedRanges:
            signature = stringToDurationSignature(key)[0]
            if duration >= signature[0] and duration <= signature[1]:
                tokens.append(value)
                tokensToDurations[value] = tokensToDurations.get(value, []) + [duration]
                break
    return tokens, tokensToDurations

def stringToDurationSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    float_arr = [float(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(float_arr), 2):
        sig.append((float_arr[i], float_arr[i + 1]))
    return sig

def sortRanges(rangesToTokens):
    return sorted(rangesToTokens.items(), key=lambda x: spanSize(stringToDurationSignature(x[0])[0]))

def spanSize(r):
    return r[1] - r[0]
#
# trafficFlowFeatures = [
#     [[True, 4444, True, 1.1, [1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1], 52, 1241], [True, 66, True, 0.2, [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 52, 1241], [True, 66, True, 10.3, [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 52, 1241], [True, 66, True, 10.6, [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 52, 1241], [True, 4444, True, 14.3, [1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1], 52, 1241]],
#     [[True, 22, True, 1.1, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241],
#      [True, 66, True, 0.7, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241],
#      [True, 1, True, 0.2, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241],
#      [True, 66, True, 1.1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241]],
# [[True, 22, True, 10.1, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241],
#      [True, 66, True, 0.007, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241],
#      [True, 1, True, 3.1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241],
#      [True, 66, True, 4.2, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 52, 1241]]
# ]
#
# trafficFlowFeatures = extractAllAaltoFeatures("devices")['D-LinkCam']
# print(trafficFlowFeatures)
# result = embed(trafficFlowFeatures, 2, 5, 4, 5)
# print(np.array(result[0]))
# print(np.array(result[1]))
# print(result[2])
# print(np.array(result[3]))
# print(result[4])
# print(np.array(result[5]))
# print(result[6])
# print(np.array(result[7]))
# print(result[8])
# print(np.array(result[9]))
# print(result[10])
# print(np.array(result[11]))
# print(result[12])
#
# print(trafficFlowFeatures)
# # print(result[0])
# reconstructedMulti = reconstruct_transformer_multivariate_complete(result[0], result[0], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12])
# # reconstructed = reconstruct_transformer_multivariate(result[0], result[0], result[3], result[7], result[9], result[11], result[5])
# # print(result[1])
# print(reconstructedMulti[5])
# generated_packet_frame_range_tokens = generate_packet_frame_range_tokens(result[0], result[0], result[1])
# generated_packet_frame_length = generate_packet_frame_length(result[0], result[0], result[1], generated_packet_frame_range_tokens, result[2], result[3])
# generated_protocol_conf = generate_protocol_conf(result[0], result[0], result[1], generated_packet_frame_range_tokens, result[2], result[3], generated_packet_frame_length, result[7])
# generated_src_ports = generate_src_ports(result[0], result[0], result[1], generated_packet_frame_range_tokens, result[2], result[3], generated_packet_frame_length, result[7], generated_protocol_conf, result[8], result[9])
# generated_dst_ports = generate_dst_ports(result[0], result[0], result[1], generated_packet_frame_range_tokens, result[2], result[3], generated_packet_frame_length, result[7], generated_protocol_conf, result[8], result[9], generated_src_ports, result[11])
# generated_duration_token = generate_duration_token(result[0], result[0], result[1], generated_packet_frame_range_tokens, result[2], result[3], generated_packet_frame_length, result[7], generated_protocol_conf, result[8], result[9], generated_src_ports, result[11], generated_dst_ports, result[5])
