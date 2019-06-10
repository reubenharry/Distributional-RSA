import csv
import pickle
from dist_rsa.utils.helperfunctions import metaphors_to_html
from dist_rsa.utils.load_data import controls,metaphors,sentences
from bs4 import BeautifulSoup
from scipy.stats import binom_test
import numpy as np
import pandas as pd

name='short_2'
a = open("html_file",'r')
soup = BeautifulSoup(a, 'html.parser')
ps = soup.find_all('p')
true_dict = {}
for p in ps:
	# print(p.input)
	# print(p.input["name"])
	# if len(p.contents[0].split(" ")) < 4: 
	# 	print(p.contents[0].split(" ")) 
	# 	raise Exception
	true_dict[p.input["name"]] = (p.contents[0], p.contents[2].replace(u'\xa0', u' ').strip(),p.contents[-1].replace(u'\xa0', u' ').strip())
# ps = [x.contents[2].replace(u'\xa0', u' ').strip() for x in ps]

l1 = pickle.load(open("dist_rsa/data/l1_dict"+name,'rb'))
baseline = pickle.load(open("dist_rsa/data/baseline_dict"+name,'rb'))

baseline[('man','lion')]=['yellow','furry']
baseline[('love','poison')]=['liquid']

def is_baseline(metaphor_list):

	metaphor = metaphor_list[0]

	metaphor = metaphor.replace(u'\xa0', u' ').strip().lower().split(" ")
	if metaphor_list[-1]=='"a1"':
		adj = metaphor_list[-3]
	elif metaphor_list[-1]=='"a2"':
		adj = metaphor_list[-2]
	else:
		print('\n\n\nfoo',metaphor_list)

		return None

	if metaphor[0] == "skin":
		metaphor = ('skin','parchment')
	else:
		metaphor = metaphor[-4],metaphor[-1]

	metaphor = metaphor[0].strip(':.'),metaphor[1].strip(':."')
	# if metaphor[1]!='parchment':
	# 	metaphor=metaphor[0],metaphor[1][:-1]
	# print("FOOOO")
	return adj in baseline[metaphor]
	
# print(ps)
# raise Exception



select_metaphors = zip(sentences,metaphors[:15])


adj_dict = metaphors_to_html(select_metaphors,l1,baseline,controls,return_dict=True)

# print(adj_dict["adj1"],adj_dict["adj2"])
answer_array = np.zeros((980))
participant_array = [0]*980

true_count=0
false_count=0
count=0
with open('dist_rsa/data/mturk_results_230917.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for i,row in enumerate(spamreader):
		if i == 0:
			adjs = [row[j][8:-1] for (j,x) in enumerate(row) if j > 27][:-2]
			print(len(adjs))
			print(adjs[48-1])
			# raise Exception

		else:

			condition0 = row[17]=='"Approved"'
			# condition1 = row[28:][35]=='"a1"'
			# condition2 = row[28:][27]=='"a1"'
			condition3 = row[28:][25]=='"a1"'
			# condition4 = row[28:][9]=='"a1"'

			# print('\n\n\nthing',row[28:][35],row[28:][27])

			conditions = condition0
			# condition3
			# condition0 and condition1 and condition2 and condition3 and condition4



			if conditions:
			# 	row[foo] == '#"Approved"'
			# 	row[foo] == bar
			# 	row[foo] == bar
			# 	row[foo] == bar
			# 	row[foo] == bar
			# count+=len(row[28:])
			# print(len(row[27:-1]))
			# print(row[28:])
			# raise Exception
			# print(row)



			# true_answers = (row[28:len(adjs)-2],i,adjs,[adj_dict[adjs[j]] for (j,x) in enumerate(adjs)])
			# true_answers = ([(adj_dict[adjs[k-1]],x) for (k,x) in enumerate(row) if k > 27 and k < len(adjs)-2])
			# print(true_answers)
				print("THING",i,[k for (k,x) in enumerate(row[28:])] )
				print([(i,x) for (i,x) in enumerate(row)])
				for k,x in enumerate(row[28:]):
						# print(k-1)
						# print(adjs[k-1])
						print(true_dict[adjs[k-1]],x,k-1,adjs[k-1])
						if (is_baseline(list(true_dict[adjs[k-1]])+[x])): 
							false_count+=1
							answer_array[count]=0
						else: 
							true_count+=1
							answer_array[count]=1
						participant_array[count]=row[16]
						count+=1

			# print([(true_dict[adjs[k-1]],x) for (k,x) in enumerate(row) if k > 27])
			# for (k,x) in enumerate(row):

			# 	if k > 27 and k < len(adjs)-2:
			# 		if x=='a1':
			# 			ans = true_dict[adjs[k-1]][1]
			# 		else:
			# 			ans = true_dict[adjs[k-1]][2]

			# 		if ans not in adj_dict:
			# 			true_count+=1
			# 		if ans in adj_dict:
			# 			false_count+=1

			else:
				print("THING",row[28:][35],row[28:][27],row[28:][25],row[28:][9])
print("true",true_count)
print("false",false_count)
print(count)
print(answer_array)
print(participant_array)

# df = pd.DataFrame({'participants':,'answers':})
# print(true_count/(20*49))
print(binom_test([true_count,false_count]))
print(binom_test([1000,1]))

# import pandas as pd
# import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold='nan') #to print the whole array


df = pd.DataFrame({'answers':answer_array, 'participants': participant_array})

model1 = sm.MixedLM.from_formula("answers ~ participants", data=df, groups=df["participants"]).fit(reml=False)
print(model1.summary())


			# print(adj_dict[adj])
			# print(row[28])
		# print(adjs)


	# return