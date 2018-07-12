#! /usr/bin/python3 -u

import sys
from os.path import exists, join
from os import makedirs
from SPARQLWrapper import SPARQLWrapper, JSON
# from multiprocessing import cpu_count, Pool as mpPool
from multiprocessing.pool import ThreadPool as Pool
import configparser as cfg


config_filename = sys.argv[1]

config = cfg.ConfigParser()
config.read(config_filename)

dir = config['DEFAULT']['directory'] + "/"

predicates_file = config['DEFAULT']['predicates_file']

########################################################################################################################

trainMap = dict()
itemMap = dict()
# featureMap = dict()

########################################################################################################################

with open(predicates_file, 'r') as file:
    predicates = [line for line in file.read().splitlines() if line and not line.startswith('#')]

########################################################################################################################

def getSPARQLResults(item):

    global itemMap
    #global features

    for predicate in predicates:

        subject = dbpediaMap[item]

        sparql = SPARQLWrapper(config['DEFAULT']['sparql_endpoint'])

        sparql.setQuery("""
            SELECT ?object
            WHERE { <""" + subject + """> <""" + predicate + """> ?object .
             FILTER(!isLiteral(?object))
             } """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            #features.add(result["object"]["value"])
            #set.add(result["object"]["value"])
            itemMap[item].add(result["object"]["value"])

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

for line in lines:
    words = line.split("\t")
    user, item, rate = [words[0], words[1], words[2]]
    if(item in dbpediaMap):
        items.add(item)

itemsList = list(items)

print(filename + " loaded. \n\tItems: {}".format(len(items)))

########################################################################################################################

for item in itemsList:
    itemMap[item] = set()

# parallel SPARQL requests

print("Fetching resources...")

p = Pool()
p.map(getSPARQLResults, itemsList)

p.close()
p.join()


print("Resources fetched.")


########################################################################################################################

if not exists(dir):
    makedirs(dir)

for item in itemMap:
    with open(join(dir, str(item)), 'w') as file:
        for feat in itemMap[item]:
            file.write("{}\n".format(feat))

########################################################################################################################

print("\nDone.")
