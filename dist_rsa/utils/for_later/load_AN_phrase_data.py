from collections import defaultdict
import pickle

def load_AN_phrase_data(metaphor=True):
	if metaphor:
		path = "dist_rsa/experiment/training_data/AN_metaphor.txt"
	else:
		path = "dist_rsa/experiment/training_data/AN_literal.txt"
	f = open(path,"r")
	return [tuple(line[:-1].split(" ")) for line in f]

def make_pair_dict():
	mets = load_AN_phrase_data(metaphor=True)
	nonmets = load_AN_phrase_data(metaphor=False)

	met_freqs = pickle.load(open("dist_rsa/experiment/training_data/met_freqs.pkl",'rb'))

	mets_adj_dict = defaultdict(list)
	mets_noun_dict = defaultdict(list)

	# nonmets_adj_dict = defaultdict(list)
	# nonmets_noun_dict = defaultdict(list)

	# print(met_freqs)
	new_mets = []
	for pair in mets:
		if met_freqs[(pair[0],pair[1])]<1:
			new_mets.append(pair)
			 
	mets = new_mets

	# print(pair[0],pair[1],met_freqs[(pair[0],pair[1])],met_freqs[(pair[0],pair[1])]<2)

	for pair in mets:
		mets_adj_dict[pair[0]] += [pair[1]]
		mets_noun_dict[pair[1]] += [pair[0]]
	# for pair in nonmets:
	# 	nonmets_adj_dict[pair[0]] += [pair[1]]
	# 	nonmets_noun_dict[pair[1]] += [pair[0]]

	return [(y,x) for (x,y) in mets][:120]

	selected_metaphors = []
	for pair in mets:

		condition1 = len(mets_adj_dict[pair[0]])>1
		condition2 = len(mets_noun_dict[pair[1]])>1

		if mets_adj_dict[pair[0]][0]==pair[1]: counter1=1
		else: counter1=0

		if mets_noun_dict[pair[1]][0]==pair[0]: counter2=1
		else: counter2=0



		if condition1 and condition2:

			selected_metaphors.append((pair))
			selected_metaphors.append((pair[0], mets_adj_dict[pair[0]][counter1]))
			selected_metaphors.append((mets_noun_dict[pair[1]][counter2], pair[1]))
			# print(pair)
			# print(pair[0], mets_adj_dict[pair[0]][counter1])
			# print(mets_noun_dict[pair[1]][counter2], pair[1])

	# remove duplicates
	def chunks(l, n):
		for i in range(0, len(l), n):
			yield l[i:i + n]

	selected_metaphors = chunks(selected_metaphors,3)

	seen_mets = []
	new_mets = []
	for triple in selected_metaphors:
		switch = True

		for met in triple:

			if met in seen_mets:
				switch = False
			
			seen_mets.append(met)

		if switch: new_mets.append([(pair[1],pair[0]) for pair in triple]) 


	return sorted(new_mets)







	return list(selected_metaphors)


	# adjs = list(set([x[0] for x in mets+nonmets]))
	# nouns = list(set([x[1] for x in mets+nonmets]))

	# for adj in adjs:
	# 	if min(len(mets_adj_dict[adj]), len(nonmets_adj_dict[adj])) > 1:
	# 		print(adj,mets_adj_dict[adj])
	# 		print(adj,nonmets_adj_dict[adj])

	# for noun in nouns:
	# 	if min(len(mets_noun_dict[noun]), len(nonmets_noun_dict[noun])) > 1:
	# 		print(noun,mets_noun_dict[noun])
	# 		print(noun,nonmets_noun_dict[noun])

