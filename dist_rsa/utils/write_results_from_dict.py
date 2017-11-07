import pickle
import numpy as np
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

			 
