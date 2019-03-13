import json
from pprint import pprint
import re
import ast
from collections import defaultdict
import csv

with open('survey.json') as f:
    data = json.load(f)


# print(len(data),len(data)-18)

def which_condition(stim):
	if bool(re.search("new_expctest1",stim)):
		return 0
	elif bool(re.search("new_expctest2",stim)):
		return 1
	elif bool(re.search("multi_expctest1",stim)):
		return 2
	else:
		return "no condition"
		# print(stim)
		# raise Exception


def trial_to_values(t):

	# print(t["data"][1])
	# print(eval(t["data"][2]["trialdata"]["responses"])["Q0"])
	# raise Exception
	# return {"condition":t["condition"],"val":int(t["data"][2]["trialdata"]["response"]),"sanity":t["data"][3]["trialdata"]["button_pressed"],"comment":t["data"][4]["trialdata"]["responses"],"true_answer":"0","url":d["data"][2]["trialdata"]["stimulus"]}
	# return {"condition":t["condition"],"val":int(t["data"][2]["trialdata"]["response"]),"sanity":t["data"][3]["trialdata"]["button_pressed"],"comment":t["data"][4]["trialdata"]["responses"],"true_answer":"2","url":d["data"][2]["trialdata"]["stimulus"]}

 	# return {"condition":t["condition"],"val":int(t["data"][3]["trialdata"]["response"]),"sanity":t["data"][2]["trialdata"]["button_pressed"],"comment":t["data"][4]["trialdata"]["responses"],"true_answer":"2","url":d["data"][3]["trialdata"]["stimulus"]}

 	# # binary phone
 	# return {"condition":t["condition"],"val":int(t["data"][2]["trialdata"]["button_pressed"]),"sanity":t["data"][3]["trialdata"]["button_pressed"],"comment":t["data"][4]["trialdata"]["responses"],"true_answer":"2","url":d["data"][2]["trialdata"]["stimulus"]}
 	
 	# binary comic
 	test_index = 3
 	sanity_1_index = 1
 	survey_index = 4
 	# print("CONDITION",t["condition"])
 	# if t["condition"]<3:
 	# print({"condition":t["condition"],"val":int(t["data"][test_index]["trialdata"]["button_pressed"]),"sanity":[t["data"][sanity_1_index]["trialdata"]["button_pressed"],],"comment":t["data"][survey_index]["trialdata"]["responses"],"true_answer":["2"],"url":d["data"][test_index]["trialdata"]["stimulus"]})
 	# print("THING",t["data"][test_index]["trialdata"])
 	# print("condition",which_condition(t["data"][test_index]["trialdata"]["stimulus"]))
 	# raise Exception
 	# return {"condition":t["condition"],"val":int(t["data"][test_index]["trialdata"]["button_pressed"]),"sanity":[t["data"][sanity_1_index]["trialdata"]["button_pressed"],t["data"][sanity_2_index]["trialdata"]["button_pressed"],],"comment":t["data"][survey_index]["trialdata"]["responses"],"true_answer":["1","2"],"url":d["data"][test_index]["trialdata"]["stimulus"]}
 	condition = which_condition(t["data"][test_index]["trialdata"]["stimulus"])
 	return {"condition":condition,"val":int(t["data"][test_index]["trialdata"]["button_pressed"]),"sanity":[t["data"][sanity_1_index]["trialdata"]["button_pressed"],],"comment":t["data"][survey_index]["trialdata"]["responses"],"true_answer":["2"],"url":d["data"][test_index]["trialdata"]["stimulus"]}
 	# except: return {"condition":t["condition"],"val":int(t["data"][5]["trialdata"]["button_pressed"]),"sanity":[t["data"][1]["trialdata"]["button_pressed"],t["data"][3]["trialdata"]["button_pressed"],t["data"][8]["trialdata"]["button_pressed"]],"comment":t["data"][6]["trialdata"]["responses"],"true_answer":["1","1","2"],"url":d["data"][5]["trialdata"]["stimulus"]}



 	# # multiparty
 	# test_index = 6
 	# sanity_1_index = 2
 	# sanity_2_index = 4
 	# survey_index = 7
 	# return {"condition":t["condition"],"val":int(t["data"][test_index]["trialdata"]["button_pressed"]),"sanity":[t["data"][sanity_1_index]["trialdata"]["button_pressed"],t["data"][sanity_2_index]["trialdata"]["button_pressed"],],"comment":t["data"][survey_index]["trialdata"]["responses"],"true_answer":["1","2"],"url":d["data"][test_index]["trialdata"]["stimulus"]}

# value = int(["data"][1]["trialdata"]["response"])

# print(len(data))

conds = {0:[],1:[],2:[]}

failures = 0
aborts = 0

cond_counter = {0:0,1:0,2:0}
cond_comment = defaultdict(list)
full_dict = defaultdict(list)

counter=0

# with open("record.txt",'w') as record:
for i in range(len(data)-288,len(data)):
# for i in list(range(495,513))+list(range(len(data)-54,len(data))):
# for i in :
	counter+=1

	try:
		d = json.loads(data[i]["datastring"])
	except: 
		aborts += 1
		continue

	# d["condition"]={""}[]

	# if d["condition"]>2:
	# 	continue

	# print(d["data"][4]["trialdata"]["responses"])
	# print(d["data"][3]["trialdata"])
	# print(d["condition"])

	# raise Exception

	result = trial_to_values(d)
	if result["condition"]=="no condition":
		continue
	choice = result["sanity"]
	print("choice",choice)

	# print("choice",choice,choice=="1")
	# print

	if choice==result["true_answer"]:
	# if True:
		conds[result["condition"]]+=[result["val"]]
		cond_counter[result["condition"]]+=1
		full_dict[result["condition"]]+=[result]
		# record.write(str(result))
		# record.write("\n\n")
		print("PING",result)
		cond_comment[result["condition"]]+=[(result["comment"],result["val"],result["url"])]
	else: failures+=1


# print(len(full_dict))
# raise Exception

print("failures",failures,"aborts",aborts,"total",counter)


print("cond counter",cond_counter)



with open('data.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

	for cond in conds:
		print("COND",cond)
		print("RAW",conds[cond],sum(conds[cond]),len(conds[cond]))
		print("NORMED",sum(conds[cond])/cond_counter[cond])

		for v in conds[cond]:
			w.writerow(['c'+str(cond),v])

print(len(data))
# with open("record.txt",'w') as record:
# 	for cond in conds:
# 		for r in full_dict[cond]:
# 			record.write(str(r))
# 			record.write('\n\n')



