import pyshark
import sys
import glob
import pickle

protocol_list = ["ARP", "LLC", "IP", "ICMP", "ICMPv6", "EAPOL", "TCP", "UDP", "HTTP", "HTTPS", "DHCP", "BOOTP", "SSDP", "DNS", "MDNS", "NTP"]

# def extractAndSaveAllAaltoFeatures(directoryOfDevices):
#     allAaltoFeatures = extractAllAaltoFeatures(directoryOfDevices)
#     for device, all_features in allAaltoFeatures.items():
#         all_real_features = []
#         max_duration = max([i[3] for i in all_features])
#         for features in all_features:
#             real_features = []
#             for featureVector in features:
#                 packet_frame_and_direction = (1 if featureVector[2] == True else -1) * featureVector[1]
#                 src_port = featureVector[5]
#                 dst_port = featureVector[6]
#                 protocol_conf = featureVector[4]
#                 normalized_duration = featureVector[3]/max_duration
#                 fV = []
#                 fV.append(packet_frame_and_direction)
#                 fV.append(src_port)
#                 fV.append(dst_port)
#                 fV += protocol_conf
#                 fV.append(normalized_duration)
#                 real_features.append(fV)
#             all_real_features.append(real_features)
#         directoryOfDevices + "/" + device + "/real_features"


def extractAllAaltoFeatures(directoryOfDevices):
    extended = directoryOfDevices + "/*"
    paths = glob.glob(extended)
    deviceToHeaders = dict()
    for directory in paths:
        directoryStr = str(directory[len(directoryOfDevices) + 1:])
        deviceToHeaders[directoryStr] = extractAaltoFeatureForDevice(directory)
    return deviceToHeaders

def extractAaltoFeatureForDevice(directoryOfDevice):
    extended = directoryOfDevice + "/"
    # path = glob.glob(extended)[0]
    pcapPath = extended + "/*.pcap"
    pcapFiles = glob.glob(pcapPath)
    features = []
    for pcapFile in pcapFiles:
        features.append(extractAaltoFeatures(pcapFile))
    return features

def extractAaltoFeatures(pathToFile):
    ipCounts = extractIpCounts(pathToFile)
    pcaps = pyshark.FileCapture(pathToFile)
    pcaps.set_debug()
    all_features = []
    idx = 0
    for packet in pcaps:
        feature = []
        feature.append(extractIsPayload(packet))
        feature.append(extractPacketFrameLength(packet))
        feature.append(extractDirection(packet, ipCounts))
        feature.append(extractTimestamp(packet))
        feature.append(extractProtocols(packet))
        feature.append(extractSrcPort(packet))
        feature.append(extractDstPort(packet))
        all_features.append(feature)
        idx += 1
    all_features.sort(key=lambda x: x[3])
    final_features = []
    previous_timestamp = all_features[0][3]
    for feature in all_features:
        new_feature = feature.copy()
        new_feature[3] = feature[3] - previous_timestamp
        previous_timestamp = feature[3]
        final_features.append(new_feature)
    return final_features

def extractProtocols(packet):
    f = []
    for protocol in protocol_list:
        if protocol in packet:
            f.append(1)
        else:
            f.append(0)
    return f

def extractIsPayload(packet):
    if 'IP' in packet and 'TCP' in packet and 'TLS' not in packet:
        return True
    else:
        if 'TLS' in packet and 'TCP' in packet and 'IP' in packet:
            try:
                tlsPCAP = getattr(packet.tls, 'tls.record.content_type')
                if tlsPCAP == 23:
                    return True
                else:
                    return False
            except:
                return False
        return False
    return False

def extractIpCounts(pathToFile):
    pcaps = pyshark.FileCapture(pathToFile)
    pcaps.set_debug()
    ipCounts = dict()
    tuples = []
    for pcap in pcaps:
        src = extractSrc(pcap)
        dst = extractDst(pcap)
        ipCounts[src] = ipCounts.get(src, 0) + 1
        ipCounts[dst] = ipCounts.get(dst, 0) + 1
    return ipCounts

def extractDirection(packet, ipCounts):
    src = extractSrc(packet)
    dst = extractDst(packet)
    if ipCounts[src] == ipCounts[dst]:
        return src < dst
    else:
        return ipCounts[src] < ipCounts[dst]  # incoming 0 outgoing 1

def extractTimestamp(packet):
    return float(packet.frame_info.time_epoch)

def extractDurations(pathToFile):
    pcaps = pyshark.FileCapture(pathToFile)
    pcaps.set_debug()
    tuples = []
    for pcap in pcaps:
        tuples.append(float(pcap.frame_info.time_epoch))
    pcaps.close()
    final_durations = []
    for i in range(len(tuples) - 1):
        final_durations.append(tuples[i + 1] - tuples[i])
    final_durations.append(0)
    return final_durations

def extractPacketFrameLength(packet):
    return int(packet.length)

def extractSrc(packet):
    if not hasattr(packet, 'tcp') and not hasattr(packet, 'udp'):
        return str(packet.eth.src)
    else:
        if hasattr(packet, 'ip'):
            return str(packet.ip.src)
        else:
            return str(packet.eth.src)

def extractDst(packet):
    if not hasattr(packet, 'tcp') and not hasattr(packet, 'udp'):
        return str(packet.eth.dst)
    else:
        if hasattr(packet, 'ip'):
            return str(packet.ip.dst)
        else:
            return str(packet.eth.dst)

def extractSrcPort(packet):
    if hasattr(packet, 'tcp'):
        return int(packet.tcp.dstport)
    if hasattr(packet, 'udp'):
        return int(packet.udp.dstport)
    return 0 # no port

def extractDstPort(packet):
    if hasattr(packet, 'tcp'):
        return int(packet.tcp.srcport)
    if hasattr(packet, 'udp'):
        return int(packet.udp.srcport)
    return 0 # no port