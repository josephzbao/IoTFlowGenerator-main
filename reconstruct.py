from random import random

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Concatenate
import numpy as np
from tensorflow.keras.utils import to_categorical

import itertools
import os
import math
import tempfile

import pytest
import numpy as np
import tensorflow as tf
from attention import Attention
import random

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
from tensorflow_addons.utils import test_utils

from keras_transformer import get_model, get_multi_output_model, decode

def reconstruct_comparison(generated_tokens, real_tokens, real_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, real_protocol_confs, tokensToProtocolConfs, real_src_ports, real_dst_ports, real_durations):
    generated_packet_frame_range_tokens = generate_packet_frame_range_tokens(generated_tokens, real_tokens, real_packet_frame_range_tokens)
    contains_sig = contains_signature(generated_packet_frame_range_tokens)
    generated_packet_frame_length = generate_packet_frame_length(generated_tokens, real_tokens, real_packet_frame_range_tokens,
                                                                 generated_packet_frame_range_tokens, tokenToSignatureRow,
                                                                 real_packet_frame_lengths)
    generated_protocol_conf = generate_protocol_conf(generated_tokens, real_tokens, real_packet_frame_range_tokens,
                                                     generated_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths,
                                                     generated_packet_frame_length, real_protocol_confs)
    generated_src_ports = generate_src_ports(generated_tokens, real_tokens, real_packet_frame_range_tokens, generated_packet_frame_range_tokens,
                                             tokenToSignatureRow, real_packet_frame_lengths, generated_packet_frame_length, real_protocol_confs,
                                             generated_protocol_conf, tokensToProtocolConfs, real_src_ports)
    generated_dst_ports = generate_dst_ports(generated_tokens, real_tokens, real_packet_frame_range_tokens, generated_packet_frame_range_tokens,
                                             tokenToSignatureRow, real_packet_frame_lengths, generated_packet_frame_length, real_protocol_confs,
                                             generated_protocol_conf, tokensToProtocolConfs, real_src_ports, generated_src_ports,
                                             real_dst_ports)
    generated_duration_token = generate_duration_token(generated_tokens, real_tokens, real_packet_frame_range_tokens, generated_packet_frame_range_tokens,
                                             tokenToSignatureRow, real_packet_frame_lengths, generated_packet_frame_length, real_protocol_confs,
                                             generated_protocol_conf, tokensToProtocolConfs, real_src_ports, generated_src_ports,
                                             real_dst_ports, generated_dst_ports, real_durations)
    return contains_sig, generated_packet_frame_length, generated_duration_token, generated_protocol_conf, generated_src_ports, generated_dst_ports

def reconstruct(generated_tokens, real_tokens, real_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, real_protocol_confs, tokensToProtocolConfs, real_src_ports, real_dst_ports, real_duration_tokens):
    generated_packet_frame_range_tokens = generate_packet_frame_range_tokens(generated_tokens, real_tokens, real_packet_frame_range_tokens)
    contains_sig = contains_signature(generated_packet_frame_range_tokens)
    generated_packet_frame_length = generate_packet_frame_length(generated_tokens, real_tokens, real_packet_frame_range_tokens,
                                                                 generated_packet_frame_range_tokens, tokenToSignatureRow,
                                                                 real_packet_frame_lengths)
    generated_protocol_conf = generate_protocol_conf(generated_tokens, real_tokens, real_packet_frame_range_tokens,
                                                     generated_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths,
                                                     generated_packet_frame_length, real_protocol_confs)
    generated_src_ports = generate_src_ports(generated_tokens, real_tokens, real_packet_frame_range_tokens, generated_packet_frame_range_tokens,
                                             tokenToSignatureRow, real_packet_frame_lengths, generated_packet_frame_length, real_protocol_confs,
                                             generated_protocol_conf, tokensToProtocolConfs, real_src_ports)
    generated_dst_ports = generate_dst_ports(generated_tokens, real_tokens, real_packet_frame_range_tokens, generated_packet_frame_range_tokens,
                                             tokenToSignatureRow, real_packet_frame_lengths, generated_packet_frame_length, real_protocol_confs,
                                             generated_protocol_conf, tokensToProtocolConfs, real_src_ports, generated_src_ports,
                                             real_dst_ports)
    generated_duration_token = generate_duration_token(generated_tokens, real_tokens, real_packet_frame_range_tokens, generated_packet_frame_range_tokens,
                                             tokenToSignatureRow, real_packet_frame_lengths, generated_packet_frame_length, real_protocol_confs, generated_protocol_conf, tokensToProtocolConfs, real_src_ports, generated_src_ports, real_dst_ports, generated_dst_ports, real_duration_tokens)
    return contains_sig, generated_packet_frame_length, generated_duration_token, generated_protocol_conf, generated_src_ports, generated_dst_ports

# def generate_packet_frame_range_tokens(generated_tokens, real_tokens, real_packet_frame_range_tokens):
#     real_tokens = np.asarray(real_tokens)
#     generated_tokens = np.asarray(generated_tokens)
#     real_packet_frame_range_tokens = np.asarray(real_packet_frame_range_tokens)
#     maxTokenValue = max(np.amax(real_tokens), np.amax(generated_tokens))
#     real_tokens = real_tokens + 2
#     generated_tokens = generated_tokens + 2
#     real_packet_frame_range_tokens = real_packet_frame_range_tokens + (maxTokenValue + 2)
#     maxPacketFrameToken = np.amax(real_packet_frame_range_tokens)
#     final_real_tokens = []
#     for tokenSequence in real_tokens:
#         newTokenSequence = tokenSequence + [1]
#         final_real_tokens.append(newTokenSequence)
#     final_generated_tokens = []
#     for tokenSequence in generated_tokens:
#         newTokenSequence = tokenSequence + [1]
#         final_generated_tokens.append(newTokenSequence)
#     train_real_packet_frame_range_tokens = []
#     for tokenSequence in real_packet_frame_range_tokens:
#         newTokenSequence = [0] + tokenSequence + [1]
#         train_real_packet_frame_range_tokens.append(newTokenSequence)
#     target_real_packet_frame_range_tokens = []
#     for tokenSequence in real_packet_frame_range_tokens:
#         newTokenSequence = tokenSequence + [1]
#         target_real_packet_frame_range_tokens.append(newTokenSequence)
#     # Build the model
#     model = get_model(
#         token_num=(maxTokenValue + maxPacketFrameToken + 2),
#         embed_dim=30,
#         encoder_num=3,
#         decoder_num=2,
#         head_num=3,
#         hidden_dim=120,
#         attention_activation='relu',
#         feed_forward_activation='relu',
#         dropout_rate=0.05,
#         embed_weights=np.random.random((maxTokenValue + maxPacketFrameToken + 2, 30)),
#     )
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#     )
#     model.summary()
#     model.fit(
#         x=[np.asarray(final_real_tokens), np.asarray(train_real_packet_frame_range_tokens)],
#         y=np.asarray(target_real_packet_frame_range_tokens),
#         epochs=5,
#     )
#     decoded = decode(
#         model,
#         np.asarray(final_generated_tokens),
#         start_token=0,
#         end_token=1,
#         pad_token=-1,
#         max_len=22,
#     )
#     decoded = np.asarray(decoded) - (maxTokenValue + 2)
#     result = []
#     for i in range(len(decoded)):
#         result.append(decoded[i][:-1])
#     return result

def mapAllElements(all_tokens, mapping):
    to_return = []
    for tokens in all_tokens:
        to_return.append(list(map(lambda x: mapping[x], tokens)))
    return to_return

def mapAllProtocolConfElements(all_tokens, mapping):
    to_return = []
    print("mapping")
    print(mapping)
    for tokens in all_tokens:
        print("protocol confs")
        print(tokens)
        to_return.append(list(map(lambda x: mapping[x], tokens)))
    return to_return


def mapDurationTokens(all_tokens, mapping):
    all_to_return = []
    for tokens in all_tokens:
        to_return = []
        for token in tokens:
            duration_set = mapping[token]
            to_return.append(random.choice(duration_set))
        all_to_return.append(to_return)
    return all_to_return

def reconstruct_transformer_multivariate_complete(generated_tokens, real_tokens, packet_frame_tokens, tokens_to_packet_frame_lengths, all_duration_tokens, all_tokens_to_durations, all_protocol_conf_tokens, tokensToProtocolConfs, all_src_port_tokens, tokensToSrcPort, all_dst_port_tokens, tokensToDstPort, all_payloads, max_duration):
    (generated_packet_frame_tokens, generated_protocol_conf_tokens, generated_src_port_tokens, generated_dst_port_tokens, generated_duration_tokens, generated_payloads) = reconstruct_transformer_multivariate(generated_tokens, real_tokens, packet_frame_tokens, all_protocol_conf_tokens, all_src_port_tokens, all_dst_port_tokens, all_duration_tokens, all_payloads)
    all_generated_packet_frame_lengths = mapAllElements(generated_packet_frame_tokens, tokens_to_packet_frame_lengths)
    all_generated_protocol_conf = mapAllProtocolConfElements(generated_protocol_conf_tokens, tokensToProtocolConfs)
    all_generated_src_ports = mapAllElements(generated_src_port_tokens, tokensToSrcPort)
    all_generated_dst_ports = mapAllElements(generated_dst_port_tokens, tokensToDstPort)
    all_generated_durations = mapDurationTokens(generated_duration_tokens, all_tokens_to_durations)
    all_generated_payloads = generated_payloads
    all_real_packet_frame_lengths = mapAllElements(packet_frame_tokens, tokens_to_packet_frame_lengths)
    all_real_protocol_confs = mapAllProtocolConfElements(all_protocol_conf_tokens, tokensToProtocolConfs)
    all_real_src_ports = mapAllElements(all_src_port_tokens, tokensToSrcPort)
    all_real_dst_ports = mapAllElements(all_dst_port_tokens, tokensToDstPort)
    all_real_durations = mapDurationTokens(all_duration_tokens, all_tokens_to_durations)
    all_real_payloads = all_payloads
    all_generated_traffic = []
    all_real_traffic = []
    print("hello")
    for i in range(len(all_generated_packet_frame_lengths)):
        generated_packet_frame_lengths = all_generated_packet_frame_lengths[i]
        generated_src_ports = all_generated_src_ports[i]
        generated_dst_ports = all_generated_dst_ports[i]
        generated_durations = all_generated_durations[i]
        generated_protocol_conf = all_generated_protocol_conf[i]
        generated_payloads = all_generated_payloads[i]
        real_packet_frame_lengths = all_real_packet_frame_lengths[i]
        real_src_ports = all_real_src_ports[i]
        real_dst_ports = all_real_dst_ports[i]
        real_durations = all_real_durations[i]
        real_protocol_confs = all_real_protocol_confs[i]
        real_payloads = all_real_payloads[i]
        generated_traffic = []
        real_traffic = []
        for j in range(len(generated_packet_frame_lengths)):
            packet = []
            packet.append(True if generated_payloads[j] == 1 else False)
            packet.append(abs(generated_packet_frame_lengths[j]))
            packet.append(generated_packet_frame_lengths[j] > 0)
            packet.append(generated_durations[j] * max_duration)
            packet.append(generated_protocol_conf[j])
            packet.append(generated_src_ports[j])
            packet.append(generated_dst_ports[j])
            generated_traffic.append(packet)
            real_packet = []
            real_packet.append(real_payloads[j])
            real_packet.append(abs(real_packet_frame_lengths[j]))
            real_packet.append(real_packet_frame_lengths[j] > 0)
            real_packet.append(real_durations[j] * max_duration)
            real_packet.append(real_protocol_confs[j])
            real_packet.append(real_src_ports[j])
            real_packet.append(real_dst_ports[j])
            real_traffic.append(real_packet)
        all_generated_traffic.append(generated_traffic)
        all_real_traffic.append(real_traffic)
    return all_generated_traffic, all_real_traffic

def reconstruct_transformer_multivariate(generated_tokens, real_tokens, real_packet_frame_length_tokens, real_protocol_confs, real_src_ports, real_dst_ports, real_duration_tokens, real_payloads):
    real_tokens = np.asarray(real_tokens)
    generated_tokens = np.asarray(generated_tokens)
    real_tokens = real_tokens + 3
    generated_tokens = generated_tokens + 3
    maxTokenValue = max(np.amax(real_tokens), np.amax(generated_tokens)) + 1
    real_packet_frame_length_tokens = np.asarray(real_packet_frame_length_tokens) + 3
    real_protocol_confs = np.asarray(real_protocol_confs) + 3
    real_src_ports = np.asarray(real_src_ports) + 3
    real_dst_ports = np.asarray(real_dst_ports) + 3
    real_duration_tokens = np.asarray(real_duration_tokens) + 3
    real_payloads = np.asarray(real_payloads) + 3
    decoder1num = np.amax(real_packet_frame_length_tokens) + 1
    decoder2num = np.amax(real_protocol_confs) + 1
    decoder3num = np.amax(real_src_ports) + 1
    decoder4num = np.amax(real_dst_ports) + 1
    decoder5num = np.amax(real_duration_tokens) + 1
    decoder6num = np.amax(real_payloads) + 1

    # Encoder training input
    final_real_tokens = []
    for tokenSequence in real_tokens:
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        final_real_tokens.append(newTokenSequence)

    # Encoder inference input
    final_generated_tokens = []
    for tokenSequence in generated_tokens:
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        final_generated_tokens.append(newTokenSequence)

    # Decoder training input
    decoder_input_tokens1 = []
    for tokenSequence in real_packet_frame_length_tokens:
        # Shift Right
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        decoder_input_tokens1.append(newTokenSequence)

    # Decoder training input
    decoder_input_tokens2 = []
    for tokenSequence in real_protocol_confs:
        # Shift Right
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        decoder_input_tokens2.append(newTokenSequence)

    # Decoder training input
    decoder_input_tokens3 = []
    for tokenSequence in real_src_ports:
        # Shift Right
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        decoder_input_tokens3.append(newTokenSequence)

    # Decoder training input
    decoder_input_tokens4 = []
    for tokenSequence in real_dst_ports:
        # Shift Right
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        decoder_input_tokens4.append(newTokenSequence)

    # Decoder training input
    decoder_input_tokens5 = []
    for tokenSequence in real_duration_tokens:
        # Shift Right
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        decoder_input_tokens5.append(newTokenSequence)

    # Decoder training input
    decoder_input_tokens6 = []
    for tokenSequence in real_payloads:
        # Shift Right
        newTokenSequence = np.append(2, tokenSequence)
        newTokenSequence = np.append(newTokenSequence, 1)
        decoder_input_tokens6.append(newTokenSequence)

    # Decoder training input
    decoder_output_tokens1 = []
    for tokenSequence in real_packet_frame_length_tokens:
        newTokenSequence = np.append(tokenSequence, [1,0])
        decoder_output_tokens1.append(newTokenSequence)

    # Decoder training input
    decoder_output_tokens2 = []
    for tokenSequence in real_protocol_confs:
        newTokenSequence = np.append(tokenSequence, [1,0])
        decoder_output_tokens2.append(newTokenSequence)

    # Decoder training input
    decoder_output_tokens3 = []
    for tokenSequence in real_src_ports:
        newTokenSequence = np.append(tokenSequence, [1,0])
        decoder_output_tokens3.append(newTokenSequence)

    # Decoder training input
    decoder_output_tokens4 = []
    for tokenSequence in real_dst_ports:
        # Shift Right
        newTokenSequence = np.append(tokenSequence, [1,0])
        decoder_output_tokens4.append(newTokenSequence)

    # Decoder training input
    decoder_output_tokens5 = []
    for tokenSequence in real_duration_tokens:
        # Shift Right
        newTokenSequence = np.append(tokenSequence, [1,0])
        decoder_output_tokens5.append(newTokenSequence)

    # Decoder training input
    decoder_output_tokens6 = []
    for tokenSequence in real_payloads:
        # Shift Right
        newTokenSequence = np.append(tokenSequence, [1, 0])
        decoder_output_tokens6.append(newTokenSequence)

    # Build the model
    model = get_multi_output_model(
        encoder_token_num=maxTokenValue,
        decoder_token_num1=decoder1num,
        decoder_token_num2=decoder2num,
        decoder_token_num3=decoder3num,
        decoder_token_num4=decoder4num,
        decoder_token_num5=decoder5num,
        decoder_token_num6=decoder6num,
        embed_dim=30,
        encoder_num=5,
        decoder_num=4,
        head_num=3,
        hidden_dim=300,
        attention_activation='relu',
        feed_forward_activation='relu',
        dropout_rate=0.05,
        encoder_embed_weights=np.random.random((maxTokenValue, 180)),
        decoder_embed_weights1=np.random.random((decoder1num, 30)),
        decoder_embed_weights2=np.random.random((decoder2num, 30)),
        decoder_embed_weights3=np.random.random((decoder3num, 30)),
        decoder_embed_weights4=np.random.random((decoder4num, 30)),
        decoder_embed_weights5=np.random.random((decoder5num, 30)),
        decoder_embed_weights6=np.random.random((decoder6num, 30)),
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
    )
    model.summary()

    # Train
    model.fit(
        x=[np.asarray(final_real_tokens), np.asarray(decoder_input_tokens1), np.asarray(decoder_input_tokens2), np.asarray(decoder_input_tokens3), np.asarray(decoder_input_tokens4), np.asarray(decoder_input_tokens5), np.asarray(decoder_input_tokens6)],
        y=[np.asarray(decoder_output_tokens1), np.asarray(decoder_output_tokens2), np.asarray(decoder_output_tokens3), np.asarray(decoder_output_tokens4), np.asarray(decoder_output_tokens5), np.asarray(decoder_output_tokens6)],
        epochs=100,
    )

    # Inference
    [decoded1, decoded2, decoded3, decoded4, decoded5, decoded6] = decode(
        model,
        final_generated_tokens,
        start_token=2,
        end_token=1,
        pad_token=0,
    )
    print("decoded1")
    print(decoded1)
    print("decoded2")
    print(decoded2)
    print("decoded3")
    print(decoded3)
    print("decoded4")
    print(decoded4)
    print("decoded5")
    print(decoded5)

    decoded1 = np.asarray(decoded1) - 3
    result1 = []
    for i in range(len(decoded1)):
        result1.append(decoded1[i][1:])

    decoded2 = np.asarray(decoded2) - 3
    result2 = []
    for i in range(len(decoded2)):
        result2.append(decoded2[i][1:])

    decoded3 = np.asarray(decoded3) - 3
    result3 = []
    for i in range(len(decoded3)):
        result3.append(decoded3[i][1:])

    decoded4 = np.asarray(decoded4) - 3
    result4 = []
    for i in range(len(decoded4)):
        result4.append(decoded4[i][1:])

    decoded5 = np.asarray(decoded5) - 3
    result5 = []
    for i in range(len(decoded5)):
        result5.append(decoded5[i][1:])

    decoded6 = np.asarray(decoded6) - 3
    result6 = []
    for i in range(len(decoded6)):
        result6.append(decoded6[i][1:])

    return (result1, result2, result3, result4, result5, result6)


def generate_transformer(generated_tokens, real_tokens, features_train):
    real_tokens = np.asarray(real_tokens)
    generated_tokens = np.asarray(generated_tokens)
    features_train = np.asarray(features_train)
    maxTokenValue = max(np.amax(real_tokens), np.amax(generated_tokens))

    # Add 2 to include start and end tokens
    real_tokens = real_tokens + 2
    generated_tokens = generated_tokens + 2

    # Disambiguate output and input tokens
    features_train = features_train + (maxTokenValue + 2)
    maxPacketFrameToken = np.amax(features_train)

    # Encoder training input
    final_real_tokens = []
    for tokenSequence in real_tokens:
        newTokenSequence = tokenSequence + [1]
        final_real_tokens.append(newTokenSequence)

    # Encoder inference input
    final_generated_tokens = []
    for tokenSequence in generated_tokens:
        newTokenSequence = tokenSequence + [1]
        final_generated_tokens.append(newTokenSequence)

    # Decoder training input
    train_x_tokens = []
    for tokenSequence in features_train:
        # Shift Right
        newTokenSequence = [0] + tokenSequence + [1]
        train_x_tokens.append(newTokenSequence)

    # Model training target
    train_y_tokens = []
    for tokenSequence in features_train:
        newTokenSequence = tokenSequence + [1]
        train_y_tokens.append(newTokenSequence)

    # Build the model
    model = get_model(
        token_num=(maxTokenValue + maxPacketFrameToken + 2),
        embed_dim=30,
        encoder_num=3,
        decoder_num=2,
        head_num=3,
        hidden_dim=120,
        attention_activation='relu',
        feed_forward_activation='relu',
        dropout_rate=0.05,
        embed_weights=np.random.random((maxTokenValue + maxPacketFrameToken + 2, 30)),
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
    )
    model.summary()

    # Train
    model.fit(
        x=[np.asarray(final_real_tokens), np.asarray(train_x_tokens)],
        y=np.asarray(train_y_tokens),
        epochs=100,
    )

    # Inference
    decoded = decode(
        model,
        np.asarray(final_generated_tokens),
        start_token=0,
        end_token=1,
        pad_token=-1,
        max_len=22,
    )

    # Normalize result
    decoded = np.asarray(decoded) - (maxTokenValue + 2)
    result = []
    for i in range(len(decoded)):
        result.append(decoded[i][:-1])
    return result

def stringtoIntArray(target):
    return list(map(lambda x: 1 if x == '1' else 0, list(target)))

def reconstructWithTransformer(generated_tokens, real_tokens, real_packet_frame_lengths, real_protocol_confs, real_src_ports, real_dst_ports, real_duration_tokens):
    reconstructedPacketFrameLengths = generate_transformer(generated_tokens, real_tokens, real_packet_frame_lengths)
    reconstructedProtocolConfs = generate_transformer(generated_tokens, real_tokens, real_protocol_confs)
    reconstructedSrcPorts = generate_transformer(generated_tokens, real_tokens, real_src_ports)
    reconstructedDstPorts = generate_transformer(generated_tokens, real_tokens, real_dst_ports)
    reconstructedDurations = generate_transformer(generated_tokens, real_tokens, real_duration_tokens)
    return (reconstructedPacketFrameLengths, reconstructedProtocolConfs, reconstructedSrcPorts, reconstructedDstPorts, reconstructedDurations)

def contains_signature(generated_packet_frame_range_tokens):
    return [[False if y == 0 else True for y in x] for x in generated_packet_frame_range_tokens]

def generate_duration_token_comparison(generated_tokens, real_tokens, real_packet_frame_range_tokens, synth_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, synth_packet_frame_lengths, real_protocol_confs, synth_protocol_confs, tokensToProtocolConfs, real_src_ports, synth_src_ports, real_dst_ports, synth_dst_ports, real_durations):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfPacketFrameTokens = int(np.amax(real_packet_frame_lengths)) + 1
    frameRangeWidth = len(list(tokenToSignatureRow.values())[0])
    protocolWidth = len(list(tokensToProtocolConfs.values())[0])
    numberOfSrcPorts = int(np.amax(real_src_ports)) + 1
    numberOfDstPorts = int(np.amax(real_dst_ports)) + 1
    model = duration_model_comparison(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens, protocolWidth, numberOfSrcPorts, numberOfDstPorts)
    signatureInput = [[tokenToSignatureRow[y] for y in x] for x in real_packet_frame_range_tokens]
    synthInput = [[tokenToSignatureRow[y] for y in x] for x in synth_packet_frame_range_tokens]
    protocolInput = [[tokensToProtocolConfs[y] for y in x] for x in real_protocol_confs]
    synthProtocolInput = [[tokensToProtocolConfs[y] for y in x] for x in synth_protocol_confs]
    model.fit([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(signatureInput), np.array(real_packet_frame_lengths), np.array(protocolInput), np.array(real_src_ports), np.array(real_dst_ports)], np.array(real_durations), batch_size=32, epochs=100)
    result = model.predict([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(synthInput), np.array(synth_packet_frame_lengths), np.array(synthProtocolInput), np.array(synth_src_ports), np.array(synth_dst_ports)])
    return result[0]

def generate_duration_token(generated_tokens, real_tokens, real_packet_frame_range_tokens, synth_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, synth_packet_frame_lengths, real_protocol_confs, synth_protocol_confs, tokensToProtocolConfs, real_src_ports, synth_src_ports, real_dst_ports, synth_dst_ports, real_duration_tokens):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfPacketFrameTokens = int(np.amax(real_packet_frame_lengths)) + 1
    frameRangeWidth = len(list(tokenToSignatureRow.values())[0])
    protocolWidth = len(list(tokensToProtocolConfs.values())[0])
    numberOfSrcPorts = int(np.amax(real_src_ports)) + 1
    numberOfDstPorts = int(np.amax(real_dst_ports)) + 1
    numberOfDurationTokens = int(np.amax(real_duration_tokens)) + 1
    model = duration_model(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens, protocolWidth, numberOfSrcPorts, numberOfDstPorts, numberOfDurationTokens)
    signatureInput = [[tokenToSignatureRow[y] for y in x] for x in real_packet_frame_range_tokens]
    synthInput = [[tokenToSignatureRow[y] for y in x] for x in synth_packet_frame_range_tokens]
    protocolInput = [[tokensToProtocolConfs[y] for y in x] for x in real_protocol_confs]
    synthProtocolInput = [[tokensToProtocolConfs[y] for y in x] for x in synth_protocol_confs]
    model.fit([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(signatureInput), np.array(real_packet_frame_lengths), np.array(protocolInput), np.array(real_src_ports), np.array(real_dst_ports)], np.array(real_duration_tokens), batch_size=32, epochs=100)
    result = model.predict([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(synthInput), np.array(synth_packet_frame_lengths), np.array(synthProtocolInput), np.array(synth_src_ports), np.array(synth_dst_ports)])
    return result[0]

def generate_dst_ports(generated_tokens, real_tokens, real_packet_frame_range_tokens, synth_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, synth_packet_frame_lengths, real_protocol_confs, synth_protocol_confs, tokensToProtocolConfs, real_src_ports, synth_src_ports, real_dst_ports):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfPacketFrameTokens = int(np.amax(real_packet_frame_lengths)) + 1
    frameRangeWidth = len(list(tokenToSignatureRow.values())[0])
    protocolWidth = len(list(tokensToProtocolConfs.values())[0])
    numberOfSrcPorts = int(np.amax(real_src_ports)) + 1
    numberOfDstPorts = int(np.amax(real_dst_ports)) + 1
    model = dst_port_model(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens, protocolWidth, numberOfSrcPorts, numberOfDstPorts)
    signatureInput = [[tokenToSignatureRow[y] for y in x] for x in real_packet_frame_range_tokens]
    synthInput = [[tokenToSignatureRow[y] for y in x] for x in synth_packet_frame_range_tokens]
    protocolInput = [[tokensToProtocolConfs[y] for y in x] for x in real_protocol_confs]
    synthProtocolInput = [[tokensToProtocolConfs[y] for y in x] for x in synth_protocol_confs]
    model.fit([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(signatureInput), np.array(real_packet_frame_lengths), np.array(protocolInput), np.array(real_src_ports)], np.array(real_dst_ports), batch_size=32, epochs=100)
    result = model.predict([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(synthInput), np.array(synth_packet_frame_lengths), np.array(synthProtocolInput), np.array(synth_src_ports)])
    return result[0]

def generate_src_ports(generated_tokens, real_tokens, real_packet_frame_range_tokens, synth_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, synth_packet_frame_lengths, real_protocol_confs, synth_protocol_confs, tokensToProtocolConfs, real_src_ports):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfPacketFrameTokens = int(np.amax(real_packet_frame_lengths)) + 1
    frameRangeWidth = len(list(tokenToSignatureRow.values())[0])
    protocolWidth = len(list(tokensToProtocolConfs.values())[0])
    numberOfSrcPorts = int(np.amax(real_src_ports)) + 1
    model = src_port_model(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens, protocolWidth, numberOfSrcPorts)
    signatureInput = [[tokenToSignatureRow[y] for y in x] for x in real_packet_frame_range_tokens]
    synthInput = [[tokenToSignatureRow[y] for y in x] for x in synth_packet_frame_range_tokens]
    protocolInput = [[tokensToProtocolConfs[y] for y in x] for x in real_protocol_confs]
    synthProtocolInput = [[tokensToProtocolConfs[y] for y in x] for x in synth_protocol_confs]
    model.fit([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(signatureInput), np.array(real_packet_frame_lengths), np.array(protocolInput)], np.array(real_src_ports), batch_size=32, epochs=100)
    result = model.predict([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(synthInput), np.array(synth_packet_frame_lengths), np.array(synthProtocolInput)])
    return result[0]

def generate_protocol_conf(generated_tokens, real_tokens, real_packet_frame_range_tokens, synth_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths, synth_packet_frame_lengths, real_protocol_confs):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfPacketFrameTokens = int(np.amax(real_packet_frame_lengths)) + 1
    frameRangeWidth = len(list(tokenToSignatureRow.values())[0])
    numberOfProtocolConfTokens = int(np.amax(real_protocol_confs)) + 1
    model = protocol_conf_model(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens, numberOfProtocolConfTokens)
    signatureInput = [[tokenToSignatureRow[y] for y in x] for x in real_packet_frame_range_tokens]
    synthInput = [[tokenToSignatureRow[y] for y in x] for x in synth_packet_frame_range_tokens]
    model.fit([np.array(np.expand_dims(np.array(real_tokens), axis=2)), np.array(signatureInput), np.array(real_packet_frame_lengths)], np.array(real_protocol_confs), batch_size=32, epochs=100)
    result = model.predict([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(synthInput), np.array(synth_packet_frame_lengths)])
    return result[0]

def generate_packet_frame_length(generated_tokens, real_tokens, real_packet_frame_range_tokens, synth_packet_frame_range_tokens, tokenToSignatureRow, real_packet_frame_lengths):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfPacketFrameTokens = int(np.amax(real_packet_frame_lengths)) + 1
    frameRangeWidth = len(list(tokenToSignatureRow.values())[0])
    model = packet_frame_length_model(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens)
    signatureInput = [[tokenToSignatureRow[y] for y in x] for x in real_packet_frame_range_tokens]
    synthInput = [[tokenToSignatureRow[y] for y in x] for x in synth_packet_frame_range_tokens]
    model.fit([np.array(np.expand_dims(np.array(real_tokens), axis=2)), np.array(signatureInput)], np.array(real_packet_frame_lengths), batch_size=32, epochs=100)
    result = model.predict([np.array(np.expand_dims(np.array(generated_tokens), axis=2)), np.array(synthInput)])
    return result[0]

def generate_packet_frame_range_tokens(generated_tokens, real_tokens, real_packet_frame_range_tokens):
    numberOfTokens = int(max(np.amax(real_tokens), np.amax(generated_tokens)))
    numberOfFrameRangeTokens = int(np.amax(real_packet_frame_range_tokens)) + 1
    model = packet_frame_range_model(numberOfTokens, numberOfFrameRangeTokens)
    model.fit(np.array(np.expand_dims(np.array(real_tokens), axis=2)), np.array(real_packet_frame_range_tokens), batch_size=32, epochs=100)
    result = model.predict(np.array(np.expand_dims(np.array(generated_tokens), axis=2)))
    return result[0]

def duration_model_comparison(numberOfTokens, frameRangeWidth, numberOfFrameLengthTokens, protocolWidth, numberOfSrcPortTokens, numberOfDstPortTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN, frameRangeWidth))
    input3 = Input(shape=(MAX_LEN,))
    input4 = Input(shape=(MAX_LEN, protocolWidth))
    input5 = Input(shape=(MAX_LEN,))
    input6 = Input(shape=(MAX_LEN,))
    embedding1 = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    embedding2 = Embedding(numberOfFrameLengthTokens + 1, 32, input_length=MAX_LEN)(input3)
    embedding3 = Embedding(numberOfSrcPortTokens + 1, 32, input_length=MAX_LEN)(input5)
    embedding4 = Embedding(numberOfDstPortTokens + 1, 32, input_length=MAX_LEN)(input6)
    merged = Concatenate(axis=2)([embedding1, embedding2, embedding3, embedding4, input2, input4])
    lstm = Bidirectional(LSTM(units=90, return_sequences=True,
                              recurrent_dropout=0.1))(merged)
    dn = TimeDistributed(Dense(90, activation="relu"))(lstm)
    dn = Dense(MAX_LEN)(dn)
    model = Model([input, input2, input3, input4, input5, input6], dn)
    model.compile(loss='mse', optimizer="adam")
    return model

def duration_model(numberOfTokens, frameRangeWidth, numberOfFrameLengthTokens, protocolWidth, numberOfSrcPortTokens, numberOfDstPortTokens, numberOfDurationTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN, frameRangeWidth))
    input3 = Input(shape=(MAX_LEN,))
    input4 = Input(shape=(MAX_LEN, protocolWidth))
    input5 = Input(shape=(MAX_LEN,))
    input6 = Input(shape=(MAX_LEN,))
    embedding1 = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    embedding2 = Embedding(numberOfFrameLengthTokens + 1, 32, input_length=MAX_LEN)(input3)
    embedding3 = Embedding(numberOfSrcPortTokens + 1, 32, input_length=MAX_LEN)(input5)
    embedding4 = Embedding(numberOfDstPortTokens + 1, 32, input_length=MAX_LEN)(input6)
    merged = Concatenate(axis=2)([embedding1, embedding2, embedding3, embedding4, input2, input4])
    merged = Attention(units=32)(merged)
    lstm = Bidirectional(LSTM(units=90, return_sequences=True,
                              recurrent_dropout=0.1))(merged)
    dn = TimeDistributed(Dense(90, activation="relu"))(lstm)
    out = CRF(numberOfDurationTokens)(dn)
    model = Model([input, input2, input3, input4, input5, input6], out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

def duration_model_ablation(numberOfTokens, numberOfDurationTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    embedding1 = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    lstm = Bidirectional(LSTM(units=90, return_sequences=True,
                              recurrent_dropout=0.1))(embedding1)
    dn = TimeDistributed(Dense(90, activation="relu"))(lstm)
    out = CRF(numberOfDurationTokens)(dn)
    model = Model([input], out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

def dst_port_model(numberOfTokens, frameRangeWidth, numberOfFrameLengthTokens, protocolWidth, numberOfSrcPortTokens, numberOfDstPortTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN, frameRangeWidth))
    input3 = Input(shape=(MAX_LEN,))
    input4 = Input(shape=(MAX_LEN, protocolWidth))
    input5 = Input(shape=(MAX_LEN,))
    embedding1 = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    embedding2 = Embedding(numberOfFrameLengthTokens + 1, 32, input_length=MAX_LEN)(input3)
    embedding3 = Embedding(numberOfSrcPortTokens + 1, 32, input_length=MAX_LEN)(input5)
    merged = Concatenate(axis=2)([embedding1, embedding2, embedding3, input2, input4])
    merged = Attention(units=32)(merged)
    lstm = Bidirectional(LSTM(units=90, return_sequences=True,
                              recurrent_dropout=0.1))(merged)
    dn = TimeDistributed(Dense(90, activation="relu"))(lstm)
    out = CRF(numberOfDstPortTokens)(dn)
    model = Model([input, input2, input3, input4, input5], out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

def src_port_model(numberOfTokens, frameRangeWidth, numberOfFrameLengthTokens, protocolWidth, numberOfSrcPortTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN, frameRangeWidth))
    input3 = Input(shape=(MAX_LEN,))
    input4 = Input(shape=(MAX_LEN, protocolWidth))
    embedding1 = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    embedding2 = Embedding(numberOfFrameLengthTokens + 1, 32, input_length=MAX_LEN)(input3)
    merged = Concatenate(axis=2)([embedding1, embedding2, input2, input4])
    merged = Attention(units=32)(merged)
    lstm = Bidirectional(LSTM(units=80, return_sequences=True,
                              recurrent_dropout=0.1))(merged)
    dn = TimeDistributed(Dense(80, activation="relu"))(lstm)
    out = CRF(numberOfSrcPortTokens)(dn)
    model = Model([input, input2, input3, input4], out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

def protocol_conf_model(numberOfTokens, frameRangeWidth, numberOfFrameLengthTokens, numberOfProtocolConfTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN, frameRangeWidth))
    input3 = Input(shape=(MAX_LEN,))
    embedding1 = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    embedding2 = Embedding(numberOfFrameLengthTokens + 1, 32, input_length=MAX_LEN)(input3)
    merged = Concatenate(axis=2)([embedding1, embedding2, input2])
    merged = Attention(units=32)(merged)
    lstm = Bidirectional(LSTM(units=70, return_sequences=True,
                              recurrent_dropout=0.1))(merged)
    dn = TimeDistributed(Dense(70, activation="relu"))(lstm)
    out = CRF(numberOfProtocolConfTokens)(dn)
    model = Model([input, input2, input3], out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

def packet_frame_length_model(numberOfTokens, frameRangeWidth, numberOfPacketFrameTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN, frameRangeWidth))
    embedding = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    merged = Concatenate(axis=2)([embedding, input2])
    lstm = Bidirectional(LSTM(units=60, return_sequences=True,
                               recurrent_dropout=0.1))(merged)
    dn = TimeDistributed(Dense(60, activation="relu"))(lstm)
    dn = Attention(units=32)(dn)
    out = CRF(numberOfPacketFrameTokens)(dn)
    model = Model([input, input2], out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

def packet_frame_range_model(numberOfTokens, numberOfFrameRangeTokens, MAX_LEN=20):
    input = Input(shape=(MAX_LEN,))
    embedding = Embedding(numberOfTokens + 1, 32, input_length=MAX_LEN)(input)
    lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(embedding)
    dn = TimeDistributed(Dense(50, activation="relu"))(lstm)
    out = CRF(numberOfFrameRangeTokens)(dn)
    model = Model(input, out)
    model = ModelWithCRFLoss(model)
    model.compile(optimizer="adam")
    return model

class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        _, potentials, sequence_length, chain_kernel = y_pred

        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        return tf.reduce_mean(crf_loss), sum(self.losses)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"crf_loss": crf_loss, "internal_losses": internal_losses}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_losses = self.compute_loss(x, y, sample_weight)
        return {"crf_loss_val": crf_loss, "internal_losses_val": internal_losses}

def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")
