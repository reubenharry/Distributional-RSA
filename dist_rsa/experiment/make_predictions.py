import scipy
import numpy as np
import pickle
import itertools
from collections import defaultdict
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.load_AN_phrase_data import load_AN_phrase_data,make_pair_dict
from dist_rsa.models.l1 import l1_model
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
import json

sig1 = 1e0
sig2 = 1e-1
l1_sig1 = 1e-1
mean_center = True
remove_top_dims = False
norm_vectors = False

path = "dist_rsa/data/results/pickles/s2memo/"
# items = [("man","shark"),("man","banana")]
vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='glove.6B.')
word_selection_vecs = vecs

def predict(LOAD,metaphors,path,hyperparams):



    # preds, quds = get_possible_utterances_and_quds(subj=subj,pred=qud,word_selection_vecs=word_selection_vecs)


    # preds = sorted(list(set(preds[:50])))
    # quds = sorted(list(set(quds[:50])))

    # hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1)

    # vec_size,vec_kind = 300,'glove.6B.'
    # vecs = load_vecs(mean=hyperparams.mean_center,pca=hyperparams.remove_top_dims,vec_length=vec_size,vec_type=vec_kind) 
    
    # for pred in preds:
    #     if pred not in vecs:
    #         # print(pred,"not in vecs")
    #         preds.remove(pred)

    if not LOAD:

        results_dict={}
    # #     for subj,pred in [("woman","rose"),("rose","woman")]:
        for subj,pred in metaphors:

            nouns,adjs = get_words(with_freqs=False)

            adj_words = [a for a in adjs if adjs[a] > concrete_threshold and a in word_selection_vecs]
            possible_utterances = sorted(adj_words,\
                key=lambda x: scipy.spatial.distance.cosine(word_selection_vecs[x],np.mean([word_selection_vecs[subj],word_selection_vecs[subj]],axis=0)),reverse=False)
            possible_utterances = sorted(possible_utterances[:100])
# adjs[n] < abstract_threshold
            qud_words = [a for a in adjs if True and a in word_selection_vecs and a not in possible_utterances]
            quds_near_subj = sorted(qud_words,\
                key=lambda x:scipy.spatial.distance.cosine(word_selection_vecs[x],word_selection_vecs[subj]),reverse=False)
            quds_near_pred = sorted(qud_words,\
                key=lambda x:scipy.spatial.distance.cosine(word_selection_vecs[x],word_selection_vecs[pred]),reverse=False)

            quds = [val for pair in zip(quds_near_subj, quds_near_pred) for val in pair]
            quds = sorted(list(set(quds[:100])))

            if pred in quds:
                quds.remove(pred) 

            for x in possible_utterances:
                if x not in vecs:
                    # print(x,"not in vecs")
                    possible_utterances.remove(x)

            for x in quds:
                if x not in vecs:
                    quds.remove(x)
            

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
                model_type=hyperparams.model_type,
                # model_type="discrete_mixture",
                calculate_projected_marginal_world_posterior=True,
                )

            

            run = Dist_RSA_Inference(params)
            run.compute_l1(load=0,save=False)
            del run
            tf.reset_default_graph()
            # print(params.marginal_means)
            results_dict[(subj,pred)]=params



            r = Results_Pickler(results_dict=results_dict,path=path+hyperparams.show())
            r.save()

    r = Results_Pickler(path=path+hyperparams.show())
    r.open()
    return r
    # print("\nHYPERPARAMS",hyperparams.show())
    # print("QUD MARGINALS",r.results_dict[metaphors[0]].qud_marginals)
    # ordered_quds = sorted(list(zip(r.results_dict[metaphors[0]].quds, r.results_dict[metaphors[0]].qud_marginals)),key=lambda x : x[1],reverse=True)[:5]
    # ordered_quds = list(zip(*ordered_quds))[0]



    # out = {"metaphor":metaphors[0],"l1":ordered_quds}
    # return json.dumps(out) 

# AN_corpus = load_AN_phrase_data()
# AN_corpus = [(subj,pred) for (pred,subj) in AN_corpus if pred in vecs and subj in vecs]

# print(AN_corpus[:2])

# model_type="numpy_discrete_mixture"
# model_type="qud_only"
# mean_center = True
# remove_top_dims = False
# norm_vectors = True
# sig1 = 1e0
# sig2 = 1e-1
# l1_sig1 = 1e-1
# hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1,model_type=model_type)
# print("HYPERPARAMS",hyperparams.show())


# out = predict(
#     LOAD=False,
#     metaphors=AN_corpus[:1],
#     path="dist_rsa/experiment/predictions/",
#     hyperparams=hyperparams)

# print(out)

def results_to_json(metaphors,LOAD):

    type_to_label = {"numpy_discrete_mixture":"l1","qud_only":"simple_l1","baseline":"baseline"}

    type_to_predictions = defaultdict(dict)

    for model_type in ["qud_only","numpy_discrete_mixture","baseline"]:

        mean_center = True
        remove_top_dims = False
        norm_vectors = False
        sig1 = 1e0
        sig2 = 1e-1
        l1_sig1 = 1e-1
        hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1,model_type=model_type)

        r = predict(
            LOAD=LOAD,
            metaphors=metaphors,
            path="dist_rsa/experiment/predictions/",
            hyperparams=hyperparams)

        # print(r.results_dict[metaphors[0]].qud_marginals)
        # raise Exception

        for metaphor in r.results_dict:

            ordered_quds = sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True)
            ordered_quds = list(list(zip(*ordered_quds))[0])
            # print("ordered_quds",ordered_quds)

            type_to_predictions[metaphor][model_type]=ordered_quds

    print(type_to_predictions)

    type_to_unique_predictions = defaultdict(dict)

    lengths = []
    for i,metaphor in enumerate(metaphors):



        l1_predictions = type_to_predictions[metaphor]["numpy_discrete_mixture"]
        qud_only_predictions = type_to_predictions[metaphor]["qud_only"]
        baseline_predictions = type_to_predictions[metaphor]["baseline"]

        lengths.append(len(list(set(l1_predictions[:2]+qud_only_predictions[:2]+baseline_predictions[:2]))))
        # # print("l1_predictions",l1_predictions)
        # removal_counter = 0
        # counter = 0
        # while counter < 2:

        #     word = l1_predictions[counter]
        #     print("WORD",word)
        #     print("elems",qud_only_predictions[:2]+baseline_predictions[:2])
        #     condition = word in qud_only_predictions[:2]+baseline_predictions[:2]
        #     if condition:
        #         print("REMOVING WORD", word)
        #         l1_predictions.remove(word)
        #         qud_only_predictions.remove(word)
        #         baseline_predictions.remove(word)
        #         removal_counter += 1
        #     else:
        #         counter+=1

        # counter = 0
        # while counter < 2:

        #     word = qud_only_predictions[counter]
        #     condition = word in baseline_predictions[:2]
        #     if condition:
        #         qud_only_predictions.remove(word)
        #         baseline_predictions.remove(word)
        #         removal_counter += 1
        #     else:
        #         counter+=1

        type_to_unique_predictions[metaphor]["numpy_discrete_mixture"] = l1_predictions[:2]
        type_to_unique_predictions[metaphor]["qud_only"] = qud_only_predictions[:2]
        type_to_unique_predictions[metaphor]["baseline"] = baseline_predictions[:2]

        # print(l1_predictions[:2]+qud_only_predictions[:2]+baseline_predictions[:2])
        # assert len(list(set(l1_predictions[:2]+qud_only_predictions[:2]+baseline_predictions[:2])))==6

    # print(type_to_unique_predictions)

    print("lengths",np.mean(lengths),len(lengths))
    output = []
    for metaphor in type_to_unique_predictions:



        new_dict = {type_to_label["numpy_discrete_mixture"]: type_to_unique_predictions[metaphor]["numpy_discrete_mixture"],
        type_to_label["qud_only"]: type_to_unique_predictions[metaphor]["qud_only"],
        type_to_label["baseline"]: type_to_unique_predictions[metaphor]["baseline"],
        "topic": metaphor[1]+" "+metaphor[0],
        "metaphor": metaphor }

        output.append(new_dict)



    # return type_to_unique_predictions
    # raise Exception
    # MAKE SURE ORDER OF METAPHORS IS FIXED: sort them
    print("output",output)
    # print("HOW MANY REMOVALS:", removal_counter)
    predictions = open("dist_rsa/json_dict","w")
    predictions.write(json.dumps(output))
    predictions.close()

AN_corpus = make_pair_dict()[:]

def join_list(l):
    return [item for sublist in l for item in sublist]

metaphors = join_list(AN_corpus)

results_to_json(metaphors=metaphors,LOAD=True)


