from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.models.l1 import l1_model

sig1_vals = [1.0,5.0,10.0,20.0]
sig2_vals = [1.0,0.5,0.1,0.05]
l1_sig1_vals = [1.0,5.0,10.0,20.0]
mean_center = [True,False]
remove_top_dims = [True,False]
norm_vectors = [True,False]

for h in itertools.product(mean_center,remove_top_dims,norm_vectors):
    mean_center = h[0]
    remove_top_dims = h[1]
    norm_vectors = h[2]
    sig1 = 20.0
    sig2 = 0.1
    l1_sig1 = 20.0
    hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1)
    print("HYPERPARAMS",hyperparams.show())


    r = Results_Pickler(path="dist_rsa/data/results/pickles/"+hyperparams.show())
    r.open()
    
    for metaphor in r.results_dict:

        print("METAPHOR:",metaphor)
        # print(sorted(list(zip(r.results_dict[metaphor].marginal_means-(r.results_dict[metaphor].subspace_prior_means[:,0]),r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True)[:5])
        # a = list(zip(r.results_dict[metaphor].quds,r.results_dict[metaphor].qud_marginals))
        # print(r.results_dict[metaphor].marginal_means.shape)
        # print(r.results_dict[metaphor].subspace_prior_means[:,0])
        # print(r.results_dict[metaphor].subspace_prior_means[:,1])
        # raise Exception

        print(sorted(list(zip(r.results_dict[metaphor].quds,r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True)[:10])
        # raise Exception
        break
        # results = dict(sorted(list(zip(r.results_dict[metaphor].quds,r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True))
        
        # print(([results[word] for word in control_set[metaphor]]))
        # break

    # for metaphor in r.results_dict:
    # 	print(metaphor,sorted(list(zip(r.results_dict[metaphor].marginal_means-r.results_dict[metaphor].subspace_prior_means,r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True)[:5])