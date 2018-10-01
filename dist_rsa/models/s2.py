from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.models.l1 import l1_model

sig1 = 1e0
sig2 = 1e-1
l1_sig1 = 1e-1
mean_center = True
remove_top_dims = False
norm_vectors = False

path = "dist_rsa/data/results/pickles/s2memo/"
# items = [("man","shark"),("man","banana")]
word_selection_vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='glove.6B.')


def s2(subj,qud,LOAD):



    preds, quds = get_possible_utterances_and_quds(subj=subj,pred=qud,word_selection_vecs=word_selection_vecs)


    preds = sorted(list(set(preds[:50])))
    quds = sorted(list(set(quds[:50])))

    hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1)

    vec_size,vec_kind = 300,'glove.6B.'
    vecs = load_vecs(mean=hyperparams.mean_center,pca=hyperparams.remove_top_dims,vec_length=vec_size,vec_type=vec_kind) 
    
    for pred in preds:
        if pred not in vecs:
            # print(pred,"not in vecs")
            preds.remove(pred)

    if not LOAD:

        results_dict={}
    #     for subj,pred in [("woman","rose"),("rose","woman")]:
        for pred in preds:
        
            possible_utterances = preds


            params = Inference_Params(
                vecs=vecs,
                subject=[subj],predicate=pred,
                quds=sorted(quds),
                possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
                sig1=hyperparams.sig1,sig2=hyperparams.sig2,l1_sig1=hyperparams.l1_sig1,
                qud_weight=0.0,freq_weight=0.0,
                number_of_qud_dimensions=1,
                poss_utt_frequencies=defaultdict(lambda:1),
                qud_frequencies=defaultdict(lambda:1),
                rationality=1.0,
                norm_vectors=hyperparams.norm_vectors,
                heatmap=False,
                resolution=Resolution(span=10,number=100),
                model_type="numpy_discrete_mixture",
                # model_type="discrete_mixture",
                # model_type="discrete_mixture",
                calculate_projected_marginal_world_posterior=True,
                )

            

            run = Dist_RSA_Inference(params)
            run.compute_l1(load=0,save=False)
            del run
            tf.reset_default_graph()
            results_dict[(subj,pred)]=params



        r = Results_Pickler(results_dict=results_dict,path=path+hyperparams.show())
        r.save()


    r = Results_Pickler(path="dist_rsa/data/results/pickles/s2memo/"+hyperparams.show())
    r.open()
    print("\nHYPERPARAMS",hyperparams.show())

    def utility(subj,pred,qud):

        results = dict(sorted(list(zip(r.results_dict[(subj,pred)].quds, r.results_dict[(subj,pred)].qud_marginals)),key=lambda x : x[-1],reverse=True))
        means = dict(sorted(list(zip(r.results_dict[(subj,pred)].quds, r.results_dict[(subj,pred)].marginal_means-r.results_dict[(subj,pred)].subspace_prior_means[:,0])),key=lambda x : x[-1],reverse=True))
        print("pred:",pred, "results:",sorted(list(zip(r.results_dict[(subj,pred)].quds, r.results_dict[(subj,pred)].marginal_means-r.results_dict[(subj,pred)].subspace_prior_means[:,0], r.results_dict[(subj,pred)].qud_marginals)),key=lambda x : x[-1],reverse=True))
        return results[qud]
        # *int(means[qud]>0)


    scores = []
    for pred in preds:
        scores.append(utility(subj=subj,pred=pred,qud=qud))

    scores = scores / np.sum(scores)

    print(sorted(list(zip(preds,scores)),key = lambda x : x[1],reverse=True))

s2(subj="man",qud="mysterious",LOAD=True)
