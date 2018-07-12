#!/usr/bin/bash

epochs=50
size=100

window=500

./word2vecWeighted_mp.py sentences.tsv features_map.dict UP/ w2vUP/ user_sentences.tsv $epochs $size $window


#####################################################

# getting and evaluating the recommendations

save="rec_pred"

# matrix=matrix.txt

python3 -u vsm.py user_sentences.tsv w2vUP/ feature_space.txt train-1m.tsv $save $1


# K="5 10 15 20 25 30 35 40 45 50 100 150 250 500 1000"
K=$(cat $1)

train=train-1m.tsv
test=test-1m.tsv

genre=genre

oracle=oracle.jar

evl="rec_eval"

for i in $K;
do
	results=$(java -jar $oracle $train $test $genre $save$i 4 10)
	echo K=$i >> $evl
	echo $results >> $evl
	echo K=$i
	echo $results
done
