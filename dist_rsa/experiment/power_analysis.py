import numpy as np
import random
import csv
from scipy.special import expit
import csv

# parameters
b = 0.7
sigma_p = 1.0
sigma_i = 1.0

NUM_PARTICIPANTS = 50
NUM_ITEMS = 120
NUM_TRIALS_PER_PARTICIPANT = 12

participant_indices = list(range(NUM_PARTICIPANTS))
item_indices = list(range(NUM_ITEMS))
participants = ["p"+str(n) for n in range(NUM_PARTICIPANTS)]
items = ['i'+str(n) for n in range(NUM_ITEMS)]

def make_rows():

	participant_intercepts = np.random.normal(0.0,sigma_p,NUM_PARTICIPANTS)
	item_intercepts = np.random.normal(0.0,sigma_i,NUM_ITEMS)

	rows = []
	for i in participant_indices:
		they_see = np.random.choice(NUM_ITEMS,NUM_TRIALS_PER_PARTICIPANT)
		for j in they_see:
			simulated_value = np.random.binomial(1,p=expit(b+participant_intercepts[i]+item_intercepts[j]))
			rows.append([participants[i],items[j],simulated_value])

	return rows

# makes n dataframes
def make_n_dfs(n):

	dfs = []
	for i in range(n):
		rows = make_rows()
		dfs.append(rows)	
	return dfs


dfs = make_n_dfs(1000)
# concatenate into one
dfs = [item for sublist in dfs for item in sublist]


with open('dist_rsa/experiment/power'+str(b)+str(sigma_p)+str(sigma_i)+'.csv', 'w', newline='') as csvfile:
    w = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in dfs:
	    w.writerow(row)


# np.random.seed(1)
# num = np.random.choice(100,3)
# print(num)
# num2 = np.random.normal(0,1,20)
# print(num2)

