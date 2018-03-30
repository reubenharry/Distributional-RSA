from collections import defaultdict
import pickle
import numpy as np
import scipy.stats
from dist_rsa.utils.load_data import animals,animal_features
from dist_rsa.models.l1_iden_1d import l1_iden_1d
from dist_rsa.utils.load_data import get_words
from collections import defaultdict
from utils.load_AN_phrase_data import load_AN_phrase_data
from dist_rsa.lm_1b_eval import predict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("load")
parser.add_argument("start_num")
parser.add_argument("stop_num")
parser.add_argument("new_dict")
args = parser.parse_args()
print(args.load,args.start_num,args.stop_num)

args.load = args.load=="True"
args.new_dict = args.new_dict=="True"
args.start_num = int(args.start_num)
args.stop_num = int(args.stop_num)

# load = sys.argv[0]
# num = sys.argv[1]
# print(sys.argv)


nouns,adjs = get_words()
concrete_threshold = 3.0
abstract_threshold = 2.5

def make_full_dict(metaphors,name):

    if args.new_dict:
        full_dict = defaultdict(list)
    else:
        full_dict = pickle.load(open("dist_rsa/data/results/pickles/full_dict_"+name,'rb'))
    for j,metaphor in enumerate(metaphors):
        metaphor = tuple(metaphor)
        subj,pred = metaphor
        # prob_dict = predict(" ".join([subj, "is"]))
        for i in range(1):
            print('\n\n\n\n',j+1,"out of",len(metaphors),"part",i,'\n\n\n\n')
            # (l1_iden_1d(metaphor))
            try:
                results = (l1_iden_1d(metaphor))
            except Exception:
                print(metaphor,"Not in dict")
                continue
            full_dict[metaphor] = results
            # gc.collect()
        # full_dict[metaphor] += [np.mean(full_dict[metaphor])]
        pickle.dump(full_dict,open("dist_rsa/data/results/pickles/full_dict_"+name,'wb'))

    return full_dict

def make_rank_dict(metaphors,name):

    if args.new_dict:
        rank_dict = defaultdict(list)
    else:
        rank_dict = pickle.load(open("dist_rsa/data/results/pickles/rank_dict_"+name,'rb'))
    for j,metaphor in enumerate(metaphors):
        metaphor = tuple(metaphor)
        subj,pred = metaphor
        # prob_dict = predict(" ".join([subj, "is"]))
        for i in range(2):
            print('\n\n\n\n',j+1,"out of",len(metaphors),"part",i,'\n\n\n\n')
            # (l1_iden_1d(metaphor))
            try:
                results = (l1_iden_1d(metaphor))
            except Exception:
                print(metaphor,"Not in dict")
                continue
            for k,result in enumerate(results):
                if result[0]==["TRIVIAL"]:
                    rank_dict[metaphor] += [k]
            # gc.collect()
        # rank_dict[metaphor] += [np.mean(rank_dict[metaphor])]
        pickle.dump(rank_dict,open("dist_rsa/data/results/pickles/rank_dict_"+name,'wb'))

    return rank_dict

def make_iden_dict(metaphors,name):

    if args.new_dict:
        iden_dict = defaultdict(list)
    else:
        iden_dict = pickle.load(open("dist_rsa/data/results/pickles/iden_dict_"+name,'rb'))
    for j,metaphor in enumerate(metaphors):
        metaphor = tuple(metaphor)
        subj,pred = metaphor
        # prob_dict = predict(" ".join([subj, "is"]))
        for i in range(2):
            print('\n\n\n\n',j+1,"out of",len(metaphors),"part",i,'\n\n\n\n')
            # (l1_iden_1d(metaphor))
            try:
                results = (l1_iden_1d(metaphor))
            except Exception:
                print(metaphor,"Not in dict")
                continue
            for result in results:
                if result[0]==["TRIVIAL"]:
                    iden_dict[metaphor] += [result[1]]
            # gc.collect()

        iden_dict[metaphor] += [np.mean(iden_dict[metaphor])]
        pickle.dump(iden_dict,open("dist_rsa/data/results/pickles/iden_dict_"+name,'wb'))

    return iden_dict

# metaphors =[('oven','fiery'),('mood','fiery')]


# make_iden_dict([('malkdjfa','alkdsjf'),('man','bat')])

# print("load=",args.load)
literal_name = "literal"
metaphorical_name = "metaphorical"

if args.load:
    print("LOADING DICTS")
    metaphorical = pickle.load(open("dist_rsa/data/results/pickles/full_dict_"+metaphorical_name,'rb'))
    literal = pickle.load(open("dist_rsa/data/results/pickles/full_dict_"+literal_name,'rb'))
    # metaphorical = pickle.load(open("dist_rsa/data/results/pickles/rank_dict_metaphorical",'rb'))
    # literal = pickle.load(open("dist_rsa/data/results/pickles/rank_dict_literal",'rb'))
    # metaphorical = pickle.load(open("dist_rsa/data/results/pickles/iden_dict_metaphorical",'rb'))
    # literal = pickle.load(open("dist_rsa/data/results/pickles/iden_dict_literal",'rb'))
else:
    print("MAKING DICTS")
    literal = make_full_dict(load_AN_phrase_data(real=False)[args.start_num:args.stop_num],literal_name)
    metaphorical = make_full_dict(load_AN_phrase_data(real=True)[args.start_num:args.stop_num],metaphorical_name)
    # literal = make_iden_dict(load_AN_phrase_data(real=False)[args.start_num:args.stop_num],"literal")
    # metaphorical = make_iden_dict(load_AN_phrase_data(real=True)[args.start_num:args.stop_num],"metaphorical")


# print(literal)
# print(metaphorical)

# lit_list = sorted(list(literal.items()),key=lambda x: x[1][-1])
# met_list = sorted(list(metaphorical.items()),key=lambda x: x[1][-1])

def trivial_density(l):
    vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.twitter.27B.pca and mean25",'rb'),encoding='latin1')
    out = []
    null_count = 0
    for j,x in enumerate(l):
        if j < 712:
            for i,y in enumerate(l[x]):
                if y[0]==["TRIVIAL"]:
                    out.append((x,y[1]))
                    null_count+=(np.exp(y[1])/scipy.spatial.distance.cosine(vecs[x[0]],vecs[x[1]]))
                    # null_count-= scipy.spatial.distance.cosine(vecs[x[0]],vecs[x[1]])
    return out,null_count

def rank(l):
    out = []
    null_count = 0
    for j,x in enumerate(l):
        if j < 712:
            for i,y in enumerate(l[x]):
                if y[0]==["TRIVIAL"]:
                    out.append((x,i))
                    if i==0:
                        null_count+=1
    return null_count

def is_metaphor(l):
    densities = trivial_density(l)[0]
    densities = [x[1] for x in densities]
    counter = 0
    for density in densities:
        print(np.exp(density))
        if np.exp(density) < 0.04:
            counter+=1
    return counter


print(trivial_density(literal)[1])
print(rank(literal))
false_pos=(is_metaphor(literal))
print("\n\n\n\n\n\nDIVIDER\n\n\n\n\n")
print(trivial_density(metaphorical)[1])
print(rank(metaphorical))
true_pos=(is_metaphor(metaphorical))

print("NUMBER OF METS/LITS",len(metaphorical),len(literal))

# print(((true_pos/len(metaphorical))+(len(literal)-false_pos)/len(literal))/2)
print((true_pos+(len(literal)-false_pos))/(711*2))

# print(how_many_literal(literal))
# print(how_many_literal(metaphorical))

    #         return i
    # else: return len(l)

# print(lit_list)
# print(met_list)

# print(how_many_metaphorical(lit_list))
# print(how_many_metaphorical(met_list))






# vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'))
# possible_utterances = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
                # key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs['man']),reverse=False)[:50]
# quds = []
# for utt in possible_utterances:
#     quds += sorted([n for n in adjs if adjs[n] < abstract_threshold and n in vecs],\
#                 key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs[utt]),reverse=False)[30:50]
# quds = list(set(quds))
# prob_dict = predict(" ".join(["men", "are",]))
# quds = sorted([n for n in adjs if adjs[n] < abstract_threshold and n in vecs and n in prob_dict],\
#     key=lambda x:prob_dict[x],reverse=True)[:200]
# print("S2 CHOSEN QUDS",quds)

# l1_cat_2d(('man','shark'))
# l1_cat_2d(('boat','knife'))
# l1_cat_3d(('man','sheep'))
# s2_cat_2d()
# s2_cat_1d(possible_utterances)

# def 
#iterate through pairs of (subj,pred) from AN corpus:
    # for each: run and save trivial_l1 in dict: quick run
    # rank by literalness score



def make_utt_to_qud_dist(possible_utterances,model,name="word_to_l1"):

    utt_to_qud_dist = {}
    for poss_utt in possible_utterances:
        utt_to_qud_dist[poss_utt] = sorted(model(("man",poss_utt),possible_utterances,quds),key=lambda x:x[0])
        pickle.dump(utt_to_qud_dist,open("dist_rsa/data/results/pickles/"+name,'wb'))

def find_rank_of_qud(q,l):
    for x in l:
        if x[0]==q:
            return x[1] 
    raise Exception(q,"not in list")

def sort_words_by_qud(utt_to_qud_dist,q):
    words = list(utt_to_qud_dist)
    return sorted(words,key=lambda x: find_rank_of_qud(q,utt_to_qud_dist[x]),reverse=True)

# make_utt_to_qud_dist(["ox","nightmare","bag"],l1_cat_2d_memo,"test_run")


# utt_to_qud_dist = pickle.load(open("dist_rsa/data/results/pickles/test_run",'rb'))
# print(memoized_s2(["independent","dumb"],utt_to_qud_dist))
