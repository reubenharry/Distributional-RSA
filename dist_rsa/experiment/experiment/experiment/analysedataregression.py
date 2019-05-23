import json
from pprint import pprint
import re
import ast
from collections import defaultdict
import csv

with open('survey.json') as f:
    data = json.load(f)

# print(data[1],len(data))
# raise Exception

NUM_OF_ITEMS = 12

met_data = [{"l1": ["ideological", "fiscal", "good"], "baseline": ["philosophical", "administrative", "greater"], "topic": "academic gap", "metaphor": ["gap", "academic"]}, {"l1": ["widespread", "alarming", "extreme"], "baseline": ["abject", "escalating", "reckless"], "topic": "acute ignorance", "metaphor": ["ignorance", "acute"]}, {"l1": ["worsening", "prevalent", "persistent"], "baseline": ["adverse", "global", "ethnic"], "topic": "acute poverty", "metaphor": ["poverty", "acute"]}, {"l1": ["timeless", "immortal", "unchanging"], "baseline": ["rejuvenated", "mellow", "unfathomable"], "topic": "ageless rhythms", "metaphor": ["rhythms", "ageless"]}, {"l1": ["ambitious", "innovative", "effective"], "baseline": ["outspoken", "risky", "academic"], "topic": "aggressive program", "metaphor": ["program", "aggressive"]}, {"l1": ["unproductive", "crummy", "stupendous"], "baseline": ["ungodly", "undoable", "mundane"], "topic": "backbreaking rent", "metaphor": ["rent", "backbreaking"]}, {"l1": ["impoverished", "prone", "poor"], "baseline": ["parallel", "important", "thrusting"], "topic": "backward area", "metaphor": ["area", "backward"]}, {"l1": ["feudal", "authoritarian", "primitive"], "baseline": ["great", "aristocratic", "founding"], "topic": "backward tradition", "metaphor": ["tradition", "backward"]}, {"l1": ["optimal", "budgetary", "vigorous"], "baseline": ["potential", "adequate", "flexible"], "topic": "balanced growth", "metaphor": ["growth", "balanced"]}, {"l1": ["unsustainable", "voluminous", "frugal"], "baseline": ["macroeconomic", "insufficient", "substantial"], "topic": "ballooning expenditure", "metaphor": ["expenditure", "ballooning"]}, {"l1": ["potential", "tremendous", "surprising"], "baseline": ["ready", "mild", "new"], "topic": "big weakness", "metaphor": ["weakness", "big"]}, {"l1": ["amusing", "cynical", "clever"], "baseline": ["nasty", "afraid", "worried"], "topic": "biting look", "metaphor": ["look", "biting"]}, {"l1": ["unrelenting", "magnificent", "majestic"], "baseline": ["windy", "torrid", "melancholic"], "topic": "blazing desolation", "metaphor": ["desolation", "blazing"]}, {"l1": ["gifted", "renowned", "younger"], "baseline": ["illiterate", "national", "affluent"], "topic": "blind elite", "metaphor": ["elite", "blind"]}, {"l1": ["evil", "suicidal", "dumb"], "baseline": ["angry", "nasty", "eyed"], "topic": "blind hate", "metaphor": ["hate", "blind"]}, {"l1": ["naive", "suicidal", "reckless"], "baseline": ["poor", "bald", "modest"], "topic": "blind optimism", "metaphor": ["optimism", "blind"]}, {"l1": ["wasteful", "unsustainable", "excess"], "baseline": ["lifeless", "tough", "modest"], "topic": "bloated spending", "metaphor": ["spending", "bloated"]}, {"l1": ["thriving", "vibrant", "ripe"], "baseline": ["pigtailed", "unraveled", "domestic"], "topic": "blossoming industry", "metaphor": ["industry", "blossoming"]}, {"l1": ["beautiful", "particular", "true"], "baseline": ["fearful", "sincere", "unrequited"], "topic": "blue feelings", "metaphor": ["feelings", "blue"]}, {"l1": ["quiet", "nondescript", "distant"], "baseline": ["forlorn", "patterned", "little"], "topic": "blue obscurity", "metaphor": ["obscurity", "blue"]}, {"l1": ["refreshing", "copious", "fresh"], "baseline": ["unopened", "divine", "artistic"], "topic": "bottled passion", "metaphor": ["passion", "bottled"]}, {"l1": ["unchecked", "continual", "unbridled"], "baseline": ["phenomenal", "zippy", "smoother"], "topic": "breakneck expansion", "metaphor": ["expansion", "breakneck"]}, {"l1": ["unable", "wrong", "serious"], "baseline": ["heavy", "anxious", "possible"], "topic": "broken hope", "metaphor": ["hope", "broken"]}, {"l1": ["minor", "whole", "similar"], "baseline": ["good", "numerous", "graceful"], "topic": "broken melody", "metaphor": ["melody", "broken"]}, {"l1": ["subsidized", "costly", "bankrupt"], "baseline": ["senior", "insolvent", "voluntary"], "topic": "burdened service", "metaphor": ["service", "burdened"]}, {"l1": ["sacred", "holy", "countless"], "baseline": ["yellow", "kindred", "crazy"], "topic": "burning soul", "metaphor": ["soul", "burning"]}, {"l1": ["disoriented", "contorted", "gaping"], "baseline": ["unexplored", "undulating", "overcrowded"], "topic": "choked gullies", "metaphor": ["gullies", "choked"]}, {"l1": ["rampant", "worsening", "prevalent"], "baseline": ["acute", "fatal", "humanitarian"], "topic": "chronic poverty", "metaphor": ["poverty", "chronic"]}, {"l1": ["moral", "secular", "sporting"], "baseline": ["parliamentary", "flexible", "artistic"], "topic": "civic fabric", "metaphor": ["fabric", "civic"]}, {"l1": ["aware", "direct", "immediate"], "baseline": ["administrative", "unable", "impossible"], "topic": "clear responsibility", "metaphor": ["responsibility", "clear"]}, {"l1": ["succinct", "emphatic", "conclusive"], "baseline": ["goddamned", "possible", "obvious"], "topic": "clear-cut answer", "metaphor": ["answer", "clear-cut"]}, {"l1": ["principled", "definite", "conclusive"], "baseline": ["lasting", "flexible", "orthographic"], "topic": "clear-cut solution", "metaphor": ["solution", "clear-cut"]}, {"l1": ["pessimistic", "bleak", "mindful"], "baseline": ["worried", "serious", "simmering"], "topic": "clouded future", "metaphor": ["future", "clouded"]}, {"l1": ["foreseeable", "pessimistic", "bleak"], "baseline": ["windless", "worrisome", "ominous"], "topic": "cloudy prospect", "metaphor": ["prospect", "cloudy"]}, {"l1": ["usual", "strange", "recent"], "baseline": ["wet", "unlikely", "mild"], "topic": "cold appearance", "metaphor": ["appearance", "cold"]}, {"l1": ["wrong", "new", "possible"], "baseline": ["warmer", "appropriate", "true"], "topic": "cold justice", "metaphor": ["justice", "cold"]}, {"l1": ["precarious", "creaking", "catastrophic"], "baseline": ["adequate", "prolonged", "eroding"], "topic": "collapsing health", "metaphor": ["health", "collapsing"]}, {"l1": ["accidental", "divergent", "disagreeable"], "baseline": ["unspoken", "intolerable", "irrefutable"], "topic": "colliding contradiction", "metaphor": ["contradiction", "colliding"]}, {"l1": ["flamboyant", "humorous", "playful"], "baseline": ["multicolored", "garish", "entertaining"], "topic": "colorful personality", "metaphor": ["personality", "colorful"]}, {"l1": ["provocative", "unflattering", "humorous"], "baseline": ["memorable", "charming", "exuberant"], "topic": "colorful remark", "metaphor": ["remark", "colorful"]}, {"l1": ["optimistic", "generous", "realistic"], "baseline": ["national", "ambitious", "virtuous"], "topic": "compassionate budget", "metaphor": ["budget", "compassionate"]}, {"l1": ["productive", "scarce", "vast"], "baseline": ["active", "hydrochloric", "needy"], "topic": "concentrated poverty", "metaphor": ["poverty", "concentrated"]}, {"l1": ["endless", "intense", "alone"], "baseline": ["dreamy", "unrelenting", "voracious"], "topic": "consuming darkness", "metaphor": ["darkness", "consuming"]}, {"l1": ["traditional", "fresh", "easy"], "baseline": ["homemade", "organizational", "pleasing"], "topic": "corporate pie", "metaphor": ["pie", "corporate"]}, {"l1": ["pervasive", "debilitating", "oppressive"], "baseline": ["addictive", "corporate", "flammable"], "topic": "corrosive corruption", "metaphor": ["corruption", "corrosive"]}, {"l1": ["dangerous", "possible", "wrong"], "baseline": ["alleged", "guilty", "logical"], "topic": "criminal path", "metaphor": ["path", "criminal"]}, {"l1": ["unbearable", "nagging", "inevitable"], "baseline": ["escalating", "playful", "deteriorating"], "topic": "crippling awkwardness", "metaphor": ["awkwardness", "crippling"]}, {"l1": ["awash", "treacherous", "countless"], "baseline": ["undulating", "unprecedented", "nightmarish"], "topic": "crisscrossed chaos", "metaphor": ["chaos", "crisscrossed"]}, {"l1": ["relentless", "debilitating", "worst"], "baseline": ["battered", "periodic", "normal"], "topic": "crushing cycle", "metaphor": ["cycle", "crushing"]}, {"l1": ["conquering", "immense", "convincing"], "baseline": ["embarrassing", "overwhelming", "actual"], "topic": "crushing difficulty", "metaphor": ["difficulty", "crushing"]}, {"l1": ["disastrous", "debilitating", "convincing"], "baseline": ["fierce", "avenging", "overwhelming"], "topic": "crushing effect", "metaphor": ["effect", "crushing"]}, {"l1": ["crippling", "debilitating", "miserable"], "baseline": ["afflicted", "nationwide", "heartbreaking"], "topic": "crushing hunger", "metaphor": ["hunger", "crushing"]}, {"l1": ["terrible", "oppressive", "miserable"], "baseline": ["absolute", "innate", "heartbreaking"], "topic": "crushing ignorance", "metaphor": ["ignorance", "crushing"]}, {"l1": ["conquering", "heartbreaking", "immense"], "baseline": ["appalling", "gigantic", "worsening"], "topic": "crushing misery", "metaphor": ["misery", "crushing"]}, {"l1": ["successive", "heartbreaking", "debilitating"], "baseline": ["aggravated", "gigantic", "ruthless"], "topic": "crushing neglect", "metaphor": ["neglect", "crushing"]}, {"l1": ["successive", "miserable", "catastrophic"], "baseline": ["fiscal", "brutal", "prolonged"], "topic": "crushing unemployment", "metaphor": ["unemployment", "crushing"]}, {"l1": ["inherent", "legitimate", "conducive"], "baseline": ["ancient", "glaring", "societal"], "topic": "cultural impediment", "metaphor": ["impediment", "cultural"]}, {"l1": ["enduring", "emotional", "psychological"], "baseline": ["aesthetic", "newfound", "geographic"], "topic": "cultural strength", "metaphor": ["strength", "cultural"]}, {"l1": ["undistorted", "faceless", "spiteful"], "baseline": ["wolfish", "pitiless", "individualistic"], "topic": "cut-throat competition", "metaphor": ["competition", "cut-throat"]}, {"l1": ["romantic", "pure", "quiet"], "baseline": ["youthful", "pale", "white"], "topic": "dark passion", "metaphor": ["passion", "dark"]}, {"l1": ["alone", "unclear", "afraid"], "baseline": ["male", "additional", "sure"], "topic": "dead money", "metaphor": ["money", "dead"]}, {"l1": ["unbearable", "mental", "chronic"], "baseline": ["afflicted", "worse", "financial"], "topic": "debilitating poverty", "metaphor": ["poverty", "debilitating"]}, {"l1": ["difficult", "obvious", "simple"], "baseline": ["preliminary", "analytical", "fresh"], "topic": "deep analysis", "metaphor": ["analysis", "deep"]}, {"l1": ["enduring", "evident", "widespread"], "baseline": ["homogeneous", "big", "prevailing"], "topic": "deep inequality", "metaphor": ["inequality", "deep"]}, {"l1": ["enduring", "rich", "inner"], "baseline": ["afflicted", "global", "underprivileged"], "topic": "deep poverty", "metaphor": ["poverty", "deep"]}, {"l1": ["certain", "other", "lower"], "baseline": ["painful", "huge", "senior"], "topic": "deep rank", "metaphor": ["rank", "deep"]}, {"l1": ["spiritual", "immense", "enduring"], "baseline": ["otherworldly", "heavy", "tremendous"], "topic": "deep solitude", "metaphor": ["solitude", "deep"]}, {"l1": ["longstanding", "unshakeable", "dualistic"], "baseline": ["baneful", "underlying", "evil"], "topic": "deep-rooted belief", "metaphor": ["belief", "deep-rooted"]}, {"l1": ["longstanding", "generational", "societal"], "baseline": ["ingrown", "hallowed", "old"], "topic": "deep-rooted tradition", "metaphor": ["tradition", "deep-rooted"]}, {"l1": ["economic", "ongoing", "renewed"], "baseline": ["bilateral", "unprecedented", "multilateral"], "topic": "deepening crisis", "metaphor": ["crisis", "deepening"]}, {"l1": ["greater", "moral", "underlying"], "baseline": ["alarming", "impoverished", "warmer"], "topic": "deeper poverty", "metaphor": ["poverty", "deeper"]}, {"l1": ["euphoric", "giddy", "overblown"], "baseline": ["despondent", "chagrined", "unruffled"], "topic": "deflated emotions", "metaphor": ["emotions", "deflated"]}, {"l1": ["overblown", "depressing", "queasy"], "baseline": ["bizarre", "depressed", "lighthearted"], "topic": "deflated joke", "metaphor": ["joke", "deflated"]}, {"l1": ["stale", "unrealistic", "waning"], "baseline": ["wobbly", "metaphorical", "certain"], "topic": "deflated meaning", "metaphor": ["meaning", "deflated"]}, {"l1": ["jubilant", "galvanized", "waning"], "baseline": ["buoyant", "wobbly", "tremendous"], "topic": "deflated pride", "metaphor": ["pride", "deflated"]}, {"l1": ["appalling", "wretched", "oppressive"], "baseline": ["alarming", "inhuman", "unromantic"], "topic": "dehumanizing poverty", "metaphor": ["poverty", "dehumanizing"]}, {"l1": ["bittersweet", "unimaginable", "unbelievable"], "baseline": ["salty", "baked", "indescribable"], "topic": "delicious agony", "metaphor": ["agony", "delicious"]}, {"l1": ["cultural", "genuine", "greater"], "baseline": ["astonishing", "parliamentary", "incumbent"], "topic": "democratic vitality", "metaphor": ["vitality", "democratic"]}, {"l1": ["quiet", "idyllic", "lonely"], "baseline": ["important", "everlasting", "secluded"], "topic": "deserted friendship", "metaphor": ["friendship", "deserted"]}, {"l1": ["lonely", "idyllic", "unnoticed"], "baseline": ["monogamous", "featureless", "enduring"], "topic": "deserted relationships", "metaphor": ["relationships", "deserted"]}, {"l1": ["serene", "otherworldly", "pristine"], "baseline": ["overgrown", "lyrical", "unforgiving"], "topic": "desolate beauty", "metaphor": ["beauty", "desolate"]}, {"l1": ["bleak", "glum", "precarious"], "baseline": ["ghastly", "enduring", "unfortunate"], "topic": "dim reminder", "metaphor": ["reminder", "dim"]}, {"l1": ["pessimistic", "bleak", "mulled"], "baseline": ["enticing", "heightened", "possible"], "topic": "dimmed prospect", "metaphor": ["prospect", "dimmed"]}, {"l1": ["wicked", "dishonest", "terrible"], "baseline": ["virtuous", "righteous", "nasty"], "topic": "dirty deeds", "metaphor": ["deeds", "dirty"]}, {"l1": ["cynical", "vulgar", "childish"], "baseline": ["lustful", "insatiable", "altruistic"], "topic": "dirty desires", "metaphor": ["desires", "dirty"]}, {"l1": ["supreme", "founding", "turbulent"], "baseline": ["constitutional", "socialist", "organic"], "topic": "dissolved time", "metaphor": ["time", "dissolved"]}, {"l1": ["sleazy", "seedy", "rowdy"], "baseline": ["new", "formal", "untrustworthy"], "topic": "dodgy bar", "metaphor": ["bar", "dodgy"]}, {"l1": ["rewarding", "excess", "vast"], "baseline": ["dry", "lavish", "potential"], "topic": "draining expense", "metaphor": ["expense", "draining"]}, {"l1": ["good", "higher", "affordable"], "baseline": ["poor", "efficient", "educated"], "topic": "durable class", "metaphor": ["class", "durable"]}, {"l1": ["desirable", "modest", "decent"], "baseline": ["residential", "fragile", "comprehensive"], "topic": "durable middle-class", "metaphor": ["middle-class", "durable"]}, {"l1": ["interactive", "competitive", "creative"], "baseline": ["nonlinear", "real", "consolidated"], "topic": "dynamic company", "metaphor": ["company", "dynamic"]}, {"l1": ["intelligent", "creative", "energetic"], "baseline": ["narcissistic", "aggressive", "likable"], "topic": "dynamic personality", "metaphor": ["personality", "dynamic"]}, {"l1": ["consequent", "structural", "fragile"], "baseline": ["apparent", "freshwater", "sustainable"], "topic": "ecological collapse", "metaphor": ["collapse", "ecological"]}, {"l1": ["civil", "disastrous", "tough"], "baseline": ["sluggish", "allied", "pivotal"], "topic": "economic battle", "metaphor": ["battle", "economic"]}, {"l1": ["immediate", "immense", "imminent"], "baseline": ["atomic", "illegal", "regional"], "topic": "economic destruction", "metaphor": ["destruction", "economic"]}, {"l1": ["scientific", "theoretical", "international"], "baseline": ["potential", "advanced", "sluggish"], "topic": "economic field", "metaphor": ["field", "economic"]}, {"l1": ["federal", "necessary", "civil"], "baseline": ["southern", "aggressive", "potential"], "topic": "economic force", "metaphor": ["force", "economic"]}, {"l1": ["stellar", "untold", "excess"], "baseline": ["humiliating", "regional", "awful"], "topic": "economic heap", "metaphor": ["heap", "economic"]}, {"l1": ["cyclical", "breakneck", "tepid"], "baseline": ["zippy", "sluggish", "unprecedented"], "topic": "economic laggard", "metaphor": ["laggard", "economic"]}, {"l1": ["academic", "practical", "basic"], "baseline": ["advanced", "recent", "other"], "topic": "economic medicine", "metaphor": ["medicine", "economic"]}, {"l1": ["imminent", "speculative", "impending"], "baseline": ["recessionary", "substantial", "agricultural"], "topic": "economic meltdown", "metaphor": ["meltdown", "economic"]}, {"l1": ["entrepreneurial", "greater", "organizational"], "baseline": ["agile", "possible", "social"], "topic": "economic mobility", "metaphor": ["mobility", "economic"]}, {"l1": ["internal", "sustained", "lower"], "baseline": ["neurological", "apparent", "robust"], "topic": "economic muscle", "metaphor": ["muscle", "economic"]}, {"l1": ["easy", "traditional", "fresh"], "baseline": ["possible", "social", "wonderful"], "topic": "economic pie", "metaphor": ["pie", "economic"]}, {"l1": ["bipartisan", "anti", "appropriate"], "baseline": ["political", "affordable", "worsening"], "topic": "economic prescription", "metaphor": ["prescription", "economic"]}, {"l1": ["ambitious", "vibrant", "civic"], "baseline": ["strategic", "postwar", "harmonious"], "topic": "economic revitalization", "metaphor": ["revitalization", "economic"]}, {"l1": ["recent", "likely", "quarterly"], "baseline": ["slight", "annual", "latest"], "topic": "economic rise", "metaphor": ["rise", "economic"]}, {"l1": ["dramatic", "depressing", "disastrous"], "baseline": ["recent", "greater", "alarming"], "topic": "economic slide", "metaphor": ["slide", "economic"]}, {"l1": ["immense", "intellectual", "geographical"], "baseline": ["overall", "analogous", "global"], "topic": "economic sphere", "metaphor": ["sphere", "economic"]}, {"l1": ["vibrant", "medieval", "multicultural"], "baseline": ["worse", "grotesque", "airy"], "topic": "economic tapestry", "metaphor": ["tapestry", "economic"]}, {"l1": ["important", "vital", "practical"], "baseline": ["potential", "automated", "greater"], "topic": "economic tool", "metaphor": ["tool", "economic"]}, {"l1": ["racial", "mutual", "ideological"], "baseline": ["relevant", "alone", "obvious"], "topic": "educational gap", "metaphor": ["gap", "educational"]}, {"l1": ["basic", "dynamic", "underlying"], "baseline": ["arbitrary", "graphical", "nonlinear"], "topic": "embedded inequality", "metaphor": ["inequality", "embedded"]}]


def trial_to_values(t):

	test_index = 3
	sanity_1_index = 1
	survey_index = 4

	values = []

	comment = t["data"][2+NUM_OF_ITEMS]["trialdata"]
	# print("COMMENT",comment)
	# raise Exception
	test = t["data"][1]["trialdata"]["response"]
	# adjectives = [test[0],test[2],test[4],test[6]]
	# vals = [test[1],test[3],test[5],test[7]]

	if not (int(test[1])>50 and int(test[3])>50 and int(test[5])<50 and int(test[7])<50):
		return None



	# adj_to_val = dict(zip(adjectives,vals))

	# print(test)

	# raise Exception

	 # >50
	 # >50
	 # <50
	 # <50


	for i in range(2,2+NUM_OF_ITEMS):


		response = t["data"][i]["trialdata"]["response"]

		participant = t['workerId']

		# print(response)
		# raise Exception



		adjectives = [response[0],response[2],response[4],response[6]]
		vals = [response[1],response[3],response[5],response[7]]

		adj_to_val = dict(zip(adjectives,vals))

		topic = t["data"][i]["trialdata"]["stimulus"][13:-4]
		# print(topic)

		# raise Exception

		# print(met_data[0]['topic'])
		# print(topic[13:-4])
		# raise Exception

		print(adjectives)
		print(vals)
		print(topic)

		metaphor = [x for x in met_data if x['topic']==topic][0]

		# print(metaphor)

		# raise Exception

		l11 = metaphor["l1"][0]
		l11val = response[response.index(l11)+1]
		l12 = metaphor["l1"][1]
		l12val = response[response.index(l12)+1]
		baseline1 = metaphor["baseline"][0]
		baseline1val = response[response.index(baseline1)+1]
		baseline2 = metaphor["baseline"][1]
		baseline2val = response[response.index(baseline2)+1]
		assert set([l11,l12,baseline1,baseline2])==set(adjectives)

		# print("bar",l11)
		# print("foo",l11val)
		# raise Exception

		# dict["l11"]=blahdata[metaphor]

		if l11val>baseline1val:
			preferred1 = 1
		elif l11val<baseline1val:
			preferred1 = 0
		else:
			preferred1 = "equal"


		if l12val>baseline2val:
			preferred2 = 1
		elif l12val<baseline2val:
			preferred2 = 0
		else:
			preferred2 = "equal"


		# values.append({})

		values.append({"baseline1":baseline1val,"baseline2":baseline2val,"l11":l11val,"l12":l12val,"metaphor":topic,"participant":participant,"comment":comment})


		# values.append({"val1":preferred1,"val2":preferred2,"metaphor":topic,"participant":participant,"comment":comment})
		# print(values)
		# raise Exception
	return values

failures = 0
aborts = 0

full_dict = defaultdict(list)

counter=0

amount_of_data = len(data)

trials = []
# with open("record.txt",'w') as record:
for i in range(len(data)-amount_of_data,len(data)):
# for i in list(range(495,513))+list(range(len(data)-54,len(data))):
# for i in :
	counter+=1

	try:
		d = json.loads(data[i]["datastring"])
	except: 
		aborts += 1
		continue

	result = trial_to_values(d)
	if result is None:
	# if False:
		failures+=1
		continue
	print("thing")
	print(result,"thing")
	# raise Exception
	# choice = result["sanity"]
	# print("choice",choice)
	trials.append(result)

	# if choice==result["true_answer"]:
	# # if True:
		
	# 	full_dict[0]+=[result]
	# 	# record.write(str(result))
	# 	# record.write("\n\n")
	# 	print("PING",result)
	# else: failures+=1


print("failures",failures,"aborts",aborts,"total",counter)
# print("full_dict",full_dict)



which = defaultdict(int)

end_counts=0
not_end_counts=0

with open('data.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

	for trial in trials:

		for result in trial:


			for x in [result["baseline1"],result["baseline2"],result["l11"],result["l12"]]:
				# print(x,"X")
				x = int(x)
				if x==100 or x==0: end_counts+=1
				else: not_end_counts+=1

			# print(result["comment"])

			w.writerow([result["metaphor"],result["participant"],"baseline", result["baseline1"]])
			w.writerow([result["metaphor"],result["participant"],"baseline", result["baseline2"]])
			w.writerow([result["metaphor"],result["participant"],"metaphor", result["l11"]])
			w.writerow([result["metaphor"],result["participant"],"metaphor", result["l12"]])
print(which)
print(end_counts,not_end_counts,"counts")

print(len(data))
