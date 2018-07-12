#! /usr/bin/python3 -u

import sys

from gensim.models import Word2Vec as word2vec
import numpy as np

from time import time

from sklearn import preprocessing

from multiprocessing import cpu_count, Pool as mpPool


senteces_file = sys.argv[1]

features_dict_file = sys.argv[2]

usersDir = sys.argv[3]

newDir = sys.argv[4]

users_sentences = sys.argv[5]

epochs = int(sys.argv[6])

size = int(sys.argv[7])

window = int(sys.argv[8])


min_count = 1

#######################################################################################

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

#######################################################################################

file = open(features_dict_file, "r")
features = file.read().splitlines()
file.close()

hashmap = dict()
reversemap = dict()
for line in features:
    index, feature = line.split("\t")
    hashmap[feature] = index
    reversemap[index] = feature

#######################################################################################

file = open(senteces_file, "r")
full_sentences = file.read().splitlines()
file.close()

weighted_feats = []

sentences = []
for item in full_sentences:
    sentence = item.split('\t')
    sentences.append(sentence)
    weighted_feats += sentence

weighted_feats = set(weighted_feats)

topn = len(weighted_feats)

#######################################################################################

print("Starting training...")

tic = time()

model = word2vec(sentences, min_count=min_count, size=size, iter=epochs, window=window)

print("\tTime: ", time()-tic)

#######################################################################################

print("Starting prediction...")

file = open(users_sentences, "r")
users = file.read().splitlines()
file.close()

dicto = dict()
for count, user in enumerate(users):
    dicto[user] = count

def getUP(user):

    file = open(usersDir + user + ".tsv", "r")
    lines = file.read().splitlines()
    file.close()

    i = dicto[user]

    all = []
    for line in lines:
        uri, w = line.split("\t")
        all.append((uri, float(w)))

    sentence = sentences[i]

    predictions = model.predict_output_word(sentence, topn=topn)  # sorted by softmax

    words = []
    for word in sentence:
        w = word.split("_")
        words.append(int(w[0]))

    uris = []
    values = []
    for prediction in predictions:
        feat = prediction[0]
        index, value = feat.split("_")
        index = int(index)

        if index not in words and index not in uris:
            uris.append(index)
            values.append(float(value))

    uris = [reversemap[str(x)] for x in uris]

    values = np.array(values).reshape(-1, 1)
    values = scaler.fit_transform(values)
    values = values.reshape(values.shape[0],)
    values = values.tolist()
    
    for i, value in enumerate(values):
        all.append((uris[i], float(value)))

    sorted(all, key=lambda x:x[1], reverse=True)

    file = open(newDir + user, "w")
    for f in all:
        file.write(f[0] + "\t" + str(f[1]) + "\n")
    file.close()

#######################################################################################

cpus = cpu_count() - 1
p = mpPool(cpus)

p.map(getUP, users)

p.close()
p.join()

#######################################################################################

print("Done.")
