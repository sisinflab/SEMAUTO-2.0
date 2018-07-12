#!/usr/bin/bash

# get User Profiles with SEMAUTO

python3 getKG.py config_KG

python3 matrix.py config_matrix

python3 semauto.py train-1m.tsv nets


########################################################################################################################

# WORD2VEC approach to augument each user's profile

epochs=50
size=100
window=500

pred="predictions"

knn_file=knns.txt

python3 word2vecWeighted.py sentences.tsv features_map.dict UP/ w2vUP/ user_sentences.tsv $epochs $size $window

python3 -u vsm.py user_sentences.tsv w2vUP/ feature_space.txt train-1m.tsv $pred $knn_file


# K="5 10 15 20 25 30 35 40 45 50 100 150 250 500 1000"
K=$(cat $knn_file)

train=train-1m.tsv
test=test-1m.tsv

genre=genre

oracle=oracle.jar

evl="evaluation"

for i in $K;
do
	results=$(java -jar $oracle $train $test $genre $pred$i 4 10)
	echo K=$i >> $evl
	echo $results >> $evl
	echo K=$i
	echo $results
done
