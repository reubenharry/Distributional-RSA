# from dist_rsa.utils.load_data import animals,animal_features
# from dist_rsa.models.l1_cat_1d import l1_cat_1d
# from dist_rsa.models.l1_cat_1d_exp import l1_cat_1d_exp
# from dist_rsa.models.l1_cat_2d import l1_cat_2d
# from dist_rsa.models.l1_cat_2d_v300 import l1_cat_2d_v300
# from dist_rsa.models.l1_cat_2d_memo import l1_cat_2d_memo
# from dist_rsa.models.l1_cat_3d import l1_cat_3d


metaphors = [("love","poison",['harmful'],['liquid']),("woman","rose",["seductive"],['floral']),("bed","heaven",["lavish"],['eternal']),
	("man","lion",["majestic"],['furry']),("room","furnace",["warm"],['metal']),("man","ox",['stubborn'],['animal']),
		("man","horse",['fast'],['patchy']),("man","tree",['stable'],['wooden']),
		("woman","lion",['ferocious'],['yellow']),("voice","river",["steady"],['wet'])]

if __name__ == "__main__":

	from dist_rsa.models.l1_cat_2d_exp import l1_cat_2d_exp
	for subj,pred,target in metaphors:
	    for i in range(2):
	        l1_cat_2d_exp((subj,pred))
        # l1_cat_1d_exp(metaphor)
