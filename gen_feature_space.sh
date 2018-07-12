#!/bin/bash
awk '{print $1}' $1/* | sort -u > feature_space.txt
