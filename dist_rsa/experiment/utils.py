import spacy

nlp = spacy.load('en')

sent = u"Another option will be to try to curb the growth in education and other local assistance , which absorbs 66 % of the state 's budget ./."

parsed = nlp(sent)

verb = [word for word in parsed if word.lemma_=="absorb"][0]
dobj = list([word for word in verb.children if word.dep_=="dobj"][0].subtree)
# print(list(dobj.subtree))

a = {"sent":parsed,"dobj":dobj}
print(a["sent"][0])

def munge(sent):
	sent = tokenizer.tokenize(sent.lower())
	without_stops = [w for w in sent if w not in stops]
	return without_stops

met_sents = [

	{
	"sent": "But in the short-term it will absorb a lot of top management 's energy and attention , '' says Philippe Haspeslagh , a business professor at the European management school , Insead , in Paris ./.",
	"subj": ["it"],
	"verb": "absorb",
	"obj": ["top","management","energy","attention"],
	},

	{
	"sent": "Shocks from one-time changes in the terms of trade should be absorbed by adjustments in exchange-rate terms -- not price levels .",
	"subj": ["shocks"],
	"verb": "absorb",
	"obj": ["adjustments","exchange","rate","terms"],
	},

	{
	"sent": "During the past 18 months , the Houston-based unit of Texas Air Corp. absorbed several carriers , in the process inheriting a hodge-podge of different aircraft .",
	"subj": ["Houston","unit","Texas","Air"],
	"verb": "absorb",
	"obj": ["several","carriers"],
	},

	{
	"sent": "In the years since 1853 , when 4 , 058 steamboats arrived at the Cincinnati waterfront loaded with `` foreign '' people and exotic merchandise , Cincinnati residents have absorbed chili and pizza and wontons and dirty rice while adjusting to electric music and jet lag .",
	"subj": ["Cincinnati","residents"],
	"verb": "absorb",
	"obj": ["chili","pizza","wontons","dirty","rice"],
	},

	# {
	# "sent": "Shocks from one-time changes in the terms of trade should be absorbed by adjustments in exchange-rate terms -- not price levels .",
	# "subj": ["shocks"],
	# "verb": "absorb",
	# "obj": ["adjustments","exchange","rate","terms"],
	# },

]


# ,

# "Mr. Wyss of Data Resources suggested that factories may be having difficulty hiring new workers at a time of such low unemployment or may simply not be able to absorb as many new workers as they would like .",

# ]





lit_sents = [

	{
	"sent": "An Energy Department spokesman says the sulfur dioxide might be simultaneously recoverable through the use of powdered limestone , which tends to absorb the sulfur .",
	"subj": ["powdered","limestone"],
	"verb": "absorb",
	"obj": ["sulfur"],
	},

	{
	"sent": "The yellow beta carotene pigment absorbs blue -LRB- not yellow -RRB- laser light .",

	"subj": ["yellow","beta","carotene","pigment"],
	"verb": "absorb",
	"obj": ["blue","laser","light"],
	},

	{
	"sent": "This time , the ground absorbed the shock waves enough to transfer her images to the metal in bas-relief .",

	"subj": ["ground"],
	"verb": "absorb",
	"obj": ["shock","waves"],
	},

	{
	"sent": "As Eliot wrote: In a warm haze , the sultry light is absorbed , not refracted , by grey stone . . . . and flowers do indeed `` sleep in empty silence .",

	"subj": ["grey","stone"],
	"verb": "absorb",
	"obj": ["sultry","light"],
	},


]
