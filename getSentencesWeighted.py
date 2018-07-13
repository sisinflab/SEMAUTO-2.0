#! /usr/bin/python3 -u

import sys
from os import listdir
from sklearn import preprocessing
import numpy as np

import configparser as cfg


config_filename = sys.argv[1]

config = cfg.ConfigParser()
config.read(config_filename)

featuresFilename = config['W2V']['features_file']

usersDir = config['SEMAUTO']['user_profiles_dir'] + '/'

features_dict_file = config['W2V']['features_dict_file']
sentences_file = config['W2V']['senteces_file']
user_sentences = config['W2V']['users_sentences']

################################################################################################

indexmap = dict()

featuresFile = open(featuresFilename, "r", encoding="utf-8")
features = featuresFile.read().splitlines()
featuresFile.close()

for index, feature in enumerate(features):
    indexmap[feature] = str(index)

file = open(features_dict_file, "w", encoding='utf-8')
for key in indexmap:
    file.write(indexmap[key] + "\t" + key + "\n")
file.close()

###############################################################################################

scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))

users = listdir(usersDir)

usersToWrite = []

sentences = open(sentences_file, "w")
users_file = open(user_sentences, "w")
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

