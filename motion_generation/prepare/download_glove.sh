#!/usr/bin/env bash
# Download the GloVe word vectors used by the text-to-motion evaluator (from MDM / HumanML3D).
# Run from motion_generation/; creates glove/our_vab_{data.npy,idx.pkl,words.pkl}.
set -e
gdown 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n -O glove.zip
rm -rf glove
unzip -q glove.zip
rm glove.zip
echo "Downloaded GloVe vectors to glove/"
