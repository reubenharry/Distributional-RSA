from sklearn.decomposition import PCA
import numpy as np
# import glove as glove
import pickle
import re



DEFAULT_FILE_PATH = "dist_rsa/data/word_vectors/"


def processVecMatrix(remove_top=False, remove_bottom=0,name="vecMatrix",remove_mean=False,vec_length=50,original_vecs="glove.6B.",tsne=False):

	number_of_vecs = 400000

	word_vectors = np.zeros((number_of_vecs, vec_length))
	words = []
	with open("dist_rsa/data/glove_texts/"+original_vecs+str(vec_length)+"d.txt") as ifs:
		for i,line in enumerate(ifs):
			line = line.strip()
			if not line:
				continue
			row = line.split()
			token = row[0]
			
			# check = [not(re.match("-?\d+?\.\d+?$", x) is None) for x in row[1:]]
			# if not(all([not(re.match("-?\d+?\.\d+?$", x) is None) for x in row[1:]])):
			# 	print("skipping line: ",i)
			# 	print(check)
			# 	print(row)
			# 	continue
			data = []
			flag = False
			for num in row[1:]:
				try:
					fl = float(num)
				except ValueError:
					print("error - not a number: ",num)
					flag = True
					fl = None
				data.append(fl)
			
			if flag:
				continue
			if len(data) != vec_length:
				print(row[1]," not included")
#				raise RuntimeError("wrong number of dimensions")
				continue
			word_vectors[i] = np.asarray(data)
			words.append(token)
			if i >= number_of_vecs-1:
				break

	# vecMatrix = dict(zip(words,word_vectors))
	# words,vecMatrix = zip(*(vecMatrix.items()))


	# print(vecMatrix)
	# vecMatrix = np.asarray(vecMatrix)

	if tsne:
		from sklearn.manifold import TSNE
		model = TSNE(n_components=2, random_state=0)
		np.set_printoptions(suppress=True)
		word_vectors = model.fit_transform(word_vectors) 
	# print('mean norm', np.linalg.norm(mean))
	# norms = np.linalg.norm(vecMatrix, axis = 1)
	# print('norm mean', np.average(norms), 'std', np.std(norms))

	if remove_mean:
		mean = np.average(word_vectors, axis = 0)
		word_vectors -= mean
	
	if remove_bottom > 0:
		pca = PCA()
		pca.fit(word_vectors)
		word_vectors -= np.dot(pca.transform(word_vectors)[:, remove_bottom:], pca.components_[remove_bottom:])

	if remove_top:

		pca = PCA()
		pca.fit(word_vectors)
		word_vectors -= np.dot(pca.transform(word_vectors)[:, :4], pca.components_[:4])	

		# return dict(zip(words,word_vectors))
	# np.save(file=name,arr=word_vectors)
	# print("saved to "+name+".npy")

	out = dict(zip(words,word_vectors))
	pickle.dump(out,open(DEFAULT_FILE_PATH+name,"wb"))
	return out

h_dict = {(False,False):'plain',(False,True):'mean_vecs',(True,False):'pca',(True,True):'pca and mean'}


if __name__ == '__main__':

	core = True
	if core:
		for pca,mean in [(False,False),(False,True),(True,False),(True,True)]:
			for number in [300,50]:
				processVecMatrix(name='glove.6B.'+h_dict[(pca,mean)]+str(number),remove_top=pca, remove_mean=mean,vec_length=number)

	# processVecMatrix(name='tsneplain2',remove_top=False, remove_mean=False,vec_length=50,tsne=True)	

	extra = True
	if extra:
		for pca,mean in [(False,False),(False,True),(True,False),(True,True)]:
			# processVecMatrix(name='glove.840B.'+h_dict[(pca,mean)]+str(300),remove_top=pca, remove_mean=mean,vec_length=300,original_vecs='glove.840B.')
			processVecMatrix(name='glove.twitter.27B.'+h_dict[(pca,mean)]+str(25),remove_top=pca, remove_mean=mean,vec_length=25,original_vecs='glove.twitter.27B.')
			processVecMatrix(name='glove.twitter.27B.'+h_dict[(pca,mean)]+str(200),remove_top=pca, remove_mean=mean,vec_length=200,original_vecs='glove.twitter.27B.')

#	processVecMatrix(name="pca__2d",remove=48, remove_mean=False,top=False,vec_length=50)



