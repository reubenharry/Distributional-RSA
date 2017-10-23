from collections import defaultdict

def load_AN_phrase_data(real=True):
	if real:
		path = "dist_rsa/data/metaphor_data/training_adj_noun_met_en.txt"
	else:
		path = "dist_rsa/data/metaphor_data/training_adj_noun_nonmet_en.txt"
	f = open(path,"r")
	return [line[:-1].split(" ") for line in f]

def make_pair_dict():
	mets = load_adj_nouns(real=True)
	nonmets = load_adj_nouns(real=False)

	mets_adj_dict = defaultdict(list)
	mets_noun_dict = defaultdict(list)

	nonmets_adj_dict = defaultdict(list)
	nonmets_noun_dict = defaultdict(list)

	for pair in mets:
		mets_adj_dict[pair[0]] += [pair[1]]
		mets_noun_dict[pair[1]] += [pair[0]]
	for pair in nonmets:
		nonmets_adj_dict[pair[0]] += [pair[1]]
		nonmets_noun_dict[pair[1]] += [pair[0]]

	adjs = list(set([x[0] for x in mets+nonmets]))
	nouns = list(set([x[1] for x in mets+nonmets]))

	for adj in adjs:
		if min(len(mets_adj_dict[adj]), len(nonmets_adj_dict[adj])) > 1:
			print(adj,mets_adj_dict[adj])
			print(adj,nonmets_adj_dict[adj])

	for noun in nouns:
		if min(len(mets_noun_dict[noun]), len(nonmets_noun_dict[noun])) > 1:
			print(noun,mets_noun_dict[noun])
			print(noun,nonmets_noun_dict[noun])

