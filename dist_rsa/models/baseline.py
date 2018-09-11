from dist_rsa.utils.load_data import load_vecs,metaphors,get_possible_utterances_and_quds


word_selection_vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='glove.6B.')

def give_qs(subj,pred):
    return sorted(list(set(get_possible_utterances_and_quds(subj=subj,pred=pred,word_selection_vecs=word_selection_vecs)[1][:100])))


for subj,pred in metaphors:
    a = input("choose")
    print("metaphor",subj,pred,str(give_qs(subj,pred)))





