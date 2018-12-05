import numpy as np
import random
import csv
from scipy.special import expit
import csv

# parameters
b = 1.0
sigma_p = 1.0
sigma_i = 1.0


participants = ["p"]*100
items = ['i']*100

participant_intercepts = np.random.normal(0.0,sigma_p,100)
item_intercepts = np.random.normal(0.0,sigma_i,100)

NUM_PARTICIPANTS = len(participants)
NUM_ITEMS = len(items)

participant_indices = list(range(NUM_PARTICIPANTS))
item_indices = list(range(NUM_ITEMS))
	
rows = []
for i in participant_indices:
	they_see = np.random.choice(NUM_ITEMS,10)
	for j in they_see:
		simulated_value = np.random.binomial(1,p=expit(b+participant_intercepts[i]+item_intercepts[j]))
		rows.append([participants[i],items[j],simulated_value])

# print(rows)

with open('dist_rsa/experiment/power.csv', 'w', newline='') as csvfile:
    w = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
	    w.writerow(row)


# np.random.seed(1)
# num = np.random.choice(100,3)
# print(num)
# num2 = np.random.normal(0,1,20)
# print(num2)

