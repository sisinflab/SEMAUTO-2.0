# SEMAUTO-2.0
Semantics-Aware Autoencoder Neural Network implemented in TensorFlow

## Paper
**Computing recommendations via a Knowledge Graph-aware Autoencoder**

Vito Bellini, Angelo Schiavone, Tommaso Di Noia, Azzurra Ragone, Eugenio Di Sciascio

### Abstract
In the last years, deep learning has shown to be a game-changing technology in artificial intelligence thanks to the numerous successes it reached in diverse application fields. Among others, the use of deep learning for the recommendation problem, although new, looks quite promising due to its positive performances in terms of accuracy of recommendation results. In a
recommendation setting, in order to predict user ratings on unknown items a possible configuration of a deep neural network is that of autoencoders tipically used to produce a lower dimensionality representation of the original
data. In this paper we present KG-AUTOENCODER, an autoencoder that bases the structure of its neural network on the semanticsaware topology of a knowledge graph thus providing a label for neurons in the hidden layer that are
eventually used to build a user profile and then compute recommendations. We show the effectiveness of KG-AUTOENCODER in terms of accuracy, diversity and novelty by comparing with state of the art recommendation algorithms.

## Introduction

SEMAUTO is a Semantics-Aware Autoencoder Neural Network which exploits knowledge encoded in a Knowledge Graph in order to build an Autoencoder whose topology reflects connections in the KG to weigh features related to rated items to build an explicit representation of the user profile.

## Configuration

All the config params are placed in config.ini file.

#### Train and test files
```
[DEFAULT]

# Train file path
training_file = train-1m.tsv

# Test file path
test_file = test-1m.tsv
```

#### Knowledge Graph settings
```
[KG]
# Mapping file
dbpedia_map = dbpediamap.tsv

# SPARQL endpoint
sparql_endpoint = http://dbpedia.org/sparql

# Directory in which to store KG data
directory = KG

# File containing a list of predicates used to explore the KG
predicates_file = predicates
```

#### SEMAUTO
```
[SEMAUTO]
# Directory in which to store matrices used by SEMAUTO
directory = nets

# Directory in which to store users profile
user_profiles_dir = UP
```

#### Word2Vec
```
[W2V]
# Voucaboulary of features
features_file = features_space.txt

# File containing a senteces made of pairs of feature and its weight (ie. feature_name-0.3)
senteces_file = sentences.tsv

# Map of indexed features
features_dict_file = features_map.dict

# Directory in which to save new enhanced users profile
newDir = w2vUP

# Keep users' order of the data listed in sentences_file
users_sentences = user_sentences.tsv

# Word2Vec model params (using gensim framework)
epochs = 50
size = 100
window = 500
min_count = 1
```

#### Item-KNN
```
[VSM]

# List of neighboorhoods size
knn_file = knns.txt

# Recommendation output file
save = predictions
```
## Run

```
$ run.sh
```

## Evaluation

We used the well know [RankSys](https://github.com/RankSys/RankSys) Java library to evaluate our recommendations.
