#! /usr/bin/python3 -u


import sys
import numpy as np
import sklearn.metrics.pairwise as skl

user_file = sys.argv[1]
user_dir = sys.argv[2]
feat_file = sys.argv[3]
train_file = sys.argv[4]
save = sys.argv[5]

knn_file = sys.argv[6]

############################################################################################################

file = open(knn_file, "r")
knns = file.read().splitlines()
file.close()

knns = [int(x) for x in knns]

############################################################################################################

print("Loading users...")

file = open(user_file, "r")
users = file.read().splitlines()
file.close()

dicto = dict()

for user in users:

    file = open(user_dir + user, "r")
    feats = file.read().splitlines()
    file.close()

    dicto[user] = []

    for feat in feats:
        uri, w = feat.split("\t")
        dicto[user].append((uri, float(w)))

#############################################################################################################

print("Loading features...")

file = open(feat_file, "r")
features = file.read().splitlines()
file.close()

index = dict()

for i, feat in enumerate(features):
    index[feat] = i

#############################################################################################################

print("Loading training file...")

file = open(train_file, "r")
lines = file.read().splitlines()
file.close()

items = set()
train = dict()

for line in lines:

    words = line.split("\t")
    user = words[0]
    item = int(words[1])
    rating = float(words[2])

    items.add(item)

    if user not in train:
        train[user] = []
    train[user].append((item, rating))

items = list(items)
it_index = dict()
for i, item in enumerate(items):
    it_index[item] = i

rating_mtx = np.zeros((len(users), len(items)))

for i, user in enumerate(users):
    its = train[user]
    for it in its:
        j = it_index[it[0]]
        rating_mtx[i, j] = it[1]

#############################################################################################################

print("Creating matrix...")

matrix = np.zeros((len(users), len(features)))

for i, user in enumerate(users):

    feats = dicto[user]

    for feat in feats:
        j = index[feat[0]]
        matrix[i, j] = feat[1]

#############################################################################################################

def getRating(similarity, matrix, rated_index):
    matrix = np.delete(matrix, rated_index, axis=1)
    weight = np.sum(np.absolute(similarity))
    ratings = np.dot(similarity, matrix)
    ratings = ratings / weight
    return ratings


def getRecs(rating_matrix, users, similar_users, items, save):

    users_dict = {}
    for index, user in enumerate(users):
        users_dict[user] = index

    recom = open(save, "w")

    for count, user in enumerate(users):

        array = np.copy(rating_matrix[count])
        rated_index = np.where(array > 0)
        unrated_movies = np.array(items)
        unrated_movies = np.delete(unrated_movies, rated_index)
        matrix = []
        similarity = []
        for similar in similar_users[user]:
            index = users_dict[similar[0]]
            similarity.append(similar[1])
            matrix.append(np.copy(rating_matrix[index]))
        matrix = np.array(matrix)
        similarity = np.array(similarity, dtype=np.float)

        ratings = getRating(similarity, matrix, rated_index)
        index = np.argsort(ratings)[-15:]
        index = index[::-1]

        for num in index:
            recom.write(user + "\t" + str(unrated_movies[num]) + "\t" + str(ratings[num]) + "\n")

    recom.close()

#############################################################################################################

similarityMatrix = skl.cosine_similarity(matrix)
np.fill_diagonal(similarityMatrix, 0)

for knn in knns:

    print("Processing", knn, "...")

    print("\tClustering...")

    clusters = dict()

    for i, user in enumerate(users):
        array = similarityMatrix[i]
        js = np.argsort(array)[-knn:]
        js = js[::-1]
        clusters[user] = []
        for j in js:
            clusters[user].append((users[j], array[j]))

    print("\tComputing recommendation...")

    getRecs(rating_mtx, users, clusters, items, save + str(knn))

#############################################################################################################

print("Done.")
