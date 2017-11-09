import pickle
import numpy as np
from dist_rsa.utils.load_data import load_vecs,get_words
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
import scipy



nouns,adjs = get_words(with_freqs=False)
vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')  

name='05-11-17_test'
out = open("dist_rsa/data/l1_results_"+name,"w")


results_dict = pickle.load(open("results_dict_"+name,'rb'))

for subj,pred in results_dict['l1_dict']:
	out.write("\n\n"+subj+" is a "+pred)

	for is_baseline in [False,True]:
		out.write('\n')
		# out.write("sig1/sig2 "+str(sig1)+"/"+str(sig2)+" baseline: "+str(is_baseline))
		if is_baseline:
			out.write('BASELINE\n')
			# results = [(0.5,0.5)]
			out.write(str([(x,np.exp(y)) for (x,y) in results_dict['qud_only_dict'][(subj,pred)]['baseline'][:5]]))
			out.write('\nBASELINE 1D:\n')
			out.write(str([(x,np.exp(y)) for (x,y) in results_dict['qud_only_dict'][(subj,pred)]['1d'][:5]]))
			out.write("\nVECTOR BASELINE\n")
			qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]
			quds = sorted(qud_words,\
			    key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
			out.write(str(quds[:5]))

		else:
			out.write("L1 RUN:\n")
			out.write("\nL1:\n")
			for i in range(3):
				out.write(str([(x,np.exp(y)) for (x,y) in results_dict['l1_dict'][(subj,pred)]['l1'+str(i)][:5]]))
				out.write('\n')
			out.write('\nDEMARG\n')
			for i in range(3):
				out.write(str([(x,np.exp(y)) for (x,y) in results_dict['l1_dict'][(subj,pred)]['demarg'+str(i)][:5]]))
				out.write('\n')
			out.write('\n1 DIMENSION\n')
			for i in range(3):
				out.write(str([(x,np.exp(y)) for (x,y) in results_dict['l1_dict'][(subj,pred)]['1d'+str(i)][:5]]))
				out.write('\n')
			out.write('\nWORLD MOVEMENT\n')
			for i in range(3):
				out.write(str([(x,np.exp(y)) for (x,y) in results_dict['l1_dict'][(subj,pred)]['world'+str(i)][:5]]))
				out.write('\n')

			 
