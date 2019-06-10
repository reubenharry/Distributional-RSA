import pickle
import numpy as np

def text_to_pickle_vecs(out_path, in_path, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = {}
    with open(in_path) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[token] = np.asarray(data)
    pickle.dump(wordVectors,open("dist_rsa/data/wordVectors"+out_path,'wb'))


text_to_pickle_vecs(out_path="50", in_path="/Users/reuben/Downloads/glove.6B/glove.6B.50d.txt",dimensions=50)