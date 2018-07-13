#!/usr/bin/bash

source <(grep = config.ini | sed 's/ *= */=/g')

# get User Profiles with SEMAUTO

python3 getKG.py config.ini

python3 matrix.py config.ini

python3 semauto.py config.ini

awk '{print $1}' $user_profiles_dir/* | sort -u > $features_file

########################################################################################################################

# WORD2VEC approach to augument each user's profile

python3 getSentencesWeighted.py config.ini

python3 word2vecWeighted.py config.ini

python3 -u vsm.py config.ini

########################################################################################################################

# Evaluation with RankSys framework

K=$(cat $knn_file)

genre=genre

oracle=oracle.jar

evl="evaluation"

for i in $K;
do
	results=$(java -jar $oracle $training_file $test_file $genre $save$i 4 10)
	echo K=$i >> $evl
	echo $results >> $evl
	echo K=$i
	echo $results
done
