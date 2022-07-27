from reconstruct import reconstruct_transformer_multivariate_complete
import glob
import pickle
import csv

devices_path = "devices"

def readTokens(path):
    with open(path, mode='r') as csvfile:
        tokens_reader = csv.reader(csvfile, delimiter=' ')
        all_tokens = []
        for row in tokens_reader:
            all_tokens.append(list(map(lambda x: int(x), row)))
        return all_tokens

extended = devices_path + "/*"
paths = glob.glob(extended)
for directory in paths:
    with open(directory + "/real_features.pkl", mode='rb') as pklfile:
        embed_result = pickle.load(pklfile)
        real_tokens = embed_result[0]
        fake_tokens = embed_result[0]
        # fake_tokens = readTokens(directory + "/generator_sample.txt")
        all_generated_traffic, all_real_traffic = reconstruct_transformer_multivariate_complete(fake_tokens, real_tokens, embed_result[3], embed_result[4], embed_result[5], embed_result[6], embed_result[7], embed_result[8], embed_result[9], embed_result[10], embed_result[11], embed_result[12], embed_result[13], embed_result[15])
        with open(directory + "/fake_device_features.pkl", mode='wb') as pklfile:
            pickle.dump(all_generated_traffic, pklfile)
            pklfile.close()
        with open(directory + "/real_device_features.pkl", mode='wb') as pklfile:
            pickle.dump(all_real_traffic, pklfile)
            pklfile.close()