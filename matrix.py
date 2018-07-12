#! /usr/bin/python3 -u

import sys
import os
from multiprocessing.pool import ThreadPool as Pool
import configparser as cfg
import numpy as np
from scipy.io import mmwrite as save
from scipy.sparse import csr_matrix as sparse

config_filename = sys.argv[1]

config = cfg.ConfigParser()
config.read(config_filename)

dir = config['DEFAULT']['directory'] + "/"

kg_dir = config['DEFAULT']['kg_dir']

########################################################################################################################

trainMap = dict()
itemMap = dict()

########################################################################################################################

def loadSPARQLResults(item):

    global itemMap
    global features

    with open(os.path.join(kg_dir, item), 'r') as file:
        results = file.read().splitlines()

    for result in results:
        features.add(result)
        #set.add(result["object"]["value"])
        itemMap[item].add(result)

########################################################################################################################

def scaledDown(value):
    x = float(value)
    max = 1
    min = 0
    A = 1
    B = 5
    x = ((((max - min) * (x - A)) / (B - A)) + min)
    return x

########################################################################################################################

class Rate:
    itemId = 0
    rating = 0

    def __init__(self, itemId, rating):
        self.itemId = itemId
        self.rating = rating

    def getRating(self):
        return self.rating

    def getItemId(self):
        return self.itemId

########################################################################################################################

print("Loading...")

# dbpedia map
filename = config['DEFAULT']['dbpedia_map']
file = open(filename, "r", encoding="utf-8")
lines = file.read().splitlines()
file.close()
dbpediaMap = dict((line.split("\t")[0], line.split("\t")[1]) for line in lines)

print(filename + " loaded.")

# training file
filename = config['DEFAULT']['training_file']
file = open(filename, "r", encoding="utf-8")
lines = file.read().splitlines()
file.close()

items = set()
users = set()

usersList = []

for line in lines:
    words = line.split("\t")
    # user, item, rate, timestamp = line.split("\t")
    user, item, rate = [words[0], words[1], words[2]]
    if(user not in trainMap):
        trainMap[user] = []
    if(item in dbpediaMap):
        trainMap[user].append(Rate(item, rate))
        items.add(item)
    users.add(user)

itemsList = list(items)

print(filename + " loaded. \n\tItems: {} \n\tUsers: {}".format(len(items), len(users)))

########################################################################################################################

for item in itemsList:
    itemMap[item] = set()

# parallel sparql

print("Fetching resources...")

features = set()

p = Pool()
p.map(loadSPARQLResults, itemsList)

p.close()
p.join()

# print(len(features))
print("Resources fetched.")

featuresList = list(features)

########################################################################################################################

print("Creating matrix...")

for user in usersList:
    rates = trainMap[user]
    items = [rate.getItemId() for rate in rates]
    ratings = np.array([scaledDown(rate.getRating()) for rate in rates])
    features = set()
    for item in items:
        features.update(itemMap[item])
    weights = np.zeros((len(items), len(features)))
    features = list(features)
    index = dict((feature, count) for count, feature in enumerate(features))

    for count, item in enumerate(items):
        row = count
        for feature in itemMap[item]:
            col = index[feature]
            weights[row, col] = 1

    dir2 = dir + str(user) + "/"

    if not(os.path.exists(dir2)):
        os.makedirs(dir2)

    save(dir2 + "mask", sparse(weights))
    save(dir2 + "matrix", sparse(ratings))

    file = open(dir2 + "features", "w")
    file.write("\n".join(features))
    file.close()

########################################################################################################################

print("\nDone.")
