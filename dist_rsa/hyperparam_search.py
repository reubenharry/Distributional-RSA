from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.models.l1 import l1_model

sig1_vals = [20.0]
# [1.0,5.0]
# [1.0,5.0,10.0,20.0]
sig2_vals = [0.1]
# [1.0,0.5,2.0]
# [1.0,0.5,0.1,0.05]
l1_sig1_vals = [20.0]
# [1.0,5.0]
# [1.0,5.0,10.0,20.0]
mean_center = [True]
remove_top_dims = [False]
norm_vectors = [True]

LOAD = False

for h in itertools.product(mean_center,remove_top_dims,norm_vectors,sig1_vals,sig2_vals,l1_sig1_vals):
    mean_center = h[0]
    remove_top_dims = h[1]
    norm_vectors = h[2]
    sig1 = h[3]
    sig2 = h[4]
    l1_sig1 = h[5]
    hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1)
    print("HYPERPARAMS",hyperparams.show())

    if not LOAD:

        results_dict={}
    #     for subj,pred in [("woman","rose"),("rose","woman")]:
        for subj,pred in control_set:
        
            results = l1_model(subj=subj,pred=pred,hyperparams=hyperparams)
            results_dict[(subj,pred)]=results

        r = Results_Pickler(results_dict=results_dict,path="dist_rsa/data/results/pickles/"+hyperparams.show())
        r.save()

    else:

        r = Results_Pickler(path="dist_rsa/data/results/pickles/"+hyperparams.show())
        r.open()
        
        score = 0
        for metaphor in r.results_dict:

            # print(sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True)[:5])
            # results = dict(sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True))
            results = sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True)
            results = [result[0] for result in results]

            results = results[:len(control_set[metaphor])]
            out = len(set(results).intersection(set(control_set[metaphor])))
            score += out
            print("METAPHOR:",metaphor)
            print(sorted(list(zip(r.results_dict[metaphor].marginal_means-(r.results_dict[metaphor].subspace_prior_means[:,0]),r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True)[:5])
            # print(out)

        print("Score", score)
            # print(([results[word] for word in control_set[metaphor]]))

            # print(sorted(list(zip(r.results_dict[metaphor].quds,r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True)[:10])
            # raise Exception
            # break

    

        # print(metaphor,sorted(list(zip(r.results_dict[metaphor].quds,r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True)[:5])