from dist_rsa.utils.helperfunctions import projection_debug
import numpy as np
from dist_rsa.tuning import metaphors
from dist_rsa.utils.load_data import load_vecs


def distance_under_projection(subj,pred,target,vecs):
	projected_subj_on_target = projection_debug(subj,np.asarray([vecs[t] for t in target]).T)
	projected_pred_on_target = projection_debug(pred,np.asarray([vecs[t] for t in target]).T)
	# projected_pred_on_target,projected_subj_on_target
	return projected_pred_on_target-projected_subj_on_target

if __name__ == "__main__":

	vecs = load_vecs(mean=True,pca=True,vec_length=25,vec_type='glove.twitter.27B.')
	for subj,pred,target,antitarget in metaphors:

		print("target distance",distance_under_projection(vecs[subj],vecs[pred],target,vecs))
		print("antitarget distance",distance_under_projection(vecs[subj],vecs[pred],antitarget,vecs))


	# projection(self.inference_params.subject_vector,np.expand_dims(vecs[word],-1))