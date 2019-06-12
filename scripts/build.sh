#make data directory
#get glove
#unzip glove
#delete zips
#put glove into ...word_vectors
mkdir -p dist_rsa/data
cd dist_rsa/data
mkdir glove_texts
mkdir results
mkdir results/pickles
cd glove_texts

wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip glove.twitter.27B.zip
rm glove.twitter.27B.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
unzip glove.6B.zip
rm glove.6B.zip

cd ../../../

pip3 install -r requirements.txt

#run refine_vectors to obtain glove_dicts

mkdir dist_rsa/data/word_vectors
ipython3 dist_rsa/utils/refine_vectors.py



