#! /usr/bin/python3 -u

import sys
from os import listdir
from sklearn import preprocessing
import numpy as np

featuresFilename = sys.argv[1]

usersDir = sys.argv[2]

################################################################################################

indexmap = dict()

featuresFile = open(featuresFilename, "r", encoding="utf-8")
features = featuresFile.read().splitlines()
featuresFile.close()

for index, feature in enumerate(features):
    indexmap[feature] = str(index)

file = open("features_map.dict", "w", encoding='utf-8')
for key in indexmap:
    file.write(indexmap[key] + "\t" + key + "\n")
file.close()

###############################################################################################

scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))

users = listdir(usersDir)

usersToWrite = []

sentences = open("sentences.tsv", "w")
users_file = open("user_sentences.tsv", "w")
for index, user in enumerate(users):

    users_file.write(user[:-4] + '\n')

    file = open(usersDir + user, "r", encoding="utf-8")
    lines = file.read().splitlines()
    file.close()

    us_feats = []
    weights = []
    for line in lines:
        feature, weight = line.split("\t")
        us_feats.append(indexmap[feature])
        weights.append(float(weight))

    weights = np.array(weights).reshape(-1, 1)
    weights = scaler.fit_transform(weights)
    weights = weights.reshape(weights.shape[0],)
    weights = weights.tolist()

    sentence = []
    for i, weight in enumerate(weights):
        w = int(weight)
        sentence.append(us_feats[i] + "_" + str(w))

    sentences.write('\t'.join(sentence) + '\n')

sentences.close()
users_file.close()

