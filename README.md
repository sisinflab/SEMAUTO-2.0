# SEMAUTO-2.0
Semantics-Aware Autoencoder Neural Network implemented in TensorFlow

## Paper
Repository related to experiments described in the paper
paper info

## Introduction

SEMAUTO is a Semantics-Aware Autoencoder Neural Network which exploits knowledge encoded in a Knowledge Graph in order to build an Autoencoder whose topology reflects connections in the KG to weigh features related to rated items to build an explicit representation of the user profile.

## Configuration

All the config params are placed in config.ini file.

Train and test files
```
[DEFAULT]

# Train file path
training_file = train-1m.tsv

# Test file path
test_file = test-1m.tsv
```

Knowledge Graph settings
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

## Evaluation

We used the well know [RankSys](https://github.com/RankSys/RankSys) Java library to evaluate our recommendations.

## Results
