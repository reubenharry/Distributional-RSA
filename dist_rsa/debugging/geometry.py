import numpy as np
import scipy
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *

vec_size,vec_kind = 300,'glove.6B.'
vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind) 

def normalize(vec):
	return vec / np.linalg.norm(vec)

normed_vicious = normalize(vecs['vicious'])
normed_swims = normalize(vecs['swims'])


def check_case(qud1,qud2,utt1,utt2):

	qud1vec = normalize(vecs[qud1])
	qud2vec = normalize(vecs[qud2])

	utt1vec = vecs[utt1]
	utt2vec = vecs[utt2]

	utt1_on_qud1 = np.dot(utt1vec,qud1vec)
	utt1_on_qud2 = np.dot(utt1vec,qud2vec)
	utt2_on_qud1 = np.dot(utt2vec,qud1vec)
	utt2_on_qud2 = np.dot(utt2vec,qud2vec)
	print("\n\n")
	print(utt1,"along",qud1, utt1_on_qud1)
	print(utt1,"along",qud2, utt1_on_qud2)
	print(utt2,"along",qud1, utt2_on_qud1)
	print(utt2,"along",qud2, utt2_on_qud2)

	print(utt1+" along "+qud1+" > "+utt2+" along "+qud1, utt1_on_qud1>utt2_on_qud1)
	print(utt2+" along "+qud2+" > "+utt1+" along "+qud2, utt2_on_qud2>utt1_on_qud2)

check_case(utt1='shark',utt2='swimmer',qud1='vicious',qud2='swims')
check_case(utt1='ox',utt2='rabbit',qud1='stubborn',qud2='timid')



# print("cosine distance of unnormalized vectors:")
# print("shark on vicious",scipy.spatial.distance.cosine(vecs['shark'],vecs['vicious']))
# print("shark on swims",scipy.spatial.distance.cosine(vecs['shark'],vecs['swims']))
# print("swimmer on vicious",scipy.spatial.distance.cosine(vecs['swimmer'],vecs['vicious']))
# print("swimmer on swims",scipy.spatial.distance.cosine(vecs['swimmer'],vecs['swims']))


