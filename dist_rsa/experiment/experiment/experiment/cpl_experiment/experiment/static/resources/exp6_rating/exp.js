
/* experiment-wide variables */
var USING_PSITURK,
	DEBUG = false,
	COUNTER = 1,
	N_TRIALS = 30,
	BASE_PAY = 0.40; //not actually used here but for reference


/* load psiturk */
try {
	var psiturk = new PsiTurk(uniqueId, adServerLoc, mode);
	USING_PSITURK = true;
} catch (exception) {
	if (exception instanceof ReferenceError) {
		console.log('Mode - running outside of psiturk');
		USING_PSITURK = false;
	} else {
		console.log('Mode - running with psiturk');
	}
};


// randomly select a subset
// stimuli = stimuli

// stimuli = getRandomSubarray(stimuli, N_TRIALS)
console.log(stimuli);

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

/* create jspsych timeline */
var timeline = [];
var data = [{"metaphor": ["policy", "backward"], "baseline": ["awkward", "clumsy", "recent"], "topic": "backward policy", "l1": ["ignorant", "inclined", "authoritarian"]}, {"metaphor": ["faith", "blind"], "baseline": ["bald", "wrong", "sincere"], "topic": "blind faith", "l1": ["misguided", "incapable", "ignorant"]}, {"metaphor": ["breeze", "fresh"], "baseline": ["sharp", "blustery", "ample"], "topic": "fresh breeze", "l1": ["steady", "vigorous", "bracing"]}, {"metaphor": ["dependency", "high"], "baseline": ["big", "recent", "functional"], "topic": "high dependency", "l1": ["greater", "excessive", "social"]}, {"metaphor": ["dreams", "empty"], "baseline": ["lifelong", "dry", "amazing"], "topic": "empty dreams", "l1": ["endless", "devoid", "meaningless"]}, {"metaphor": ["start", "fresh"], "baseline": ["ample", "upcoming", "bottled"], "topic": "fresh start", "l1": ["ready", "new", "latest"]}, {"metaphor": ["desolation", "blazing"], "baseline": ["inconsolable", "exquisite", "pristine"], "topic": "blazing desolation", "l1": ["dashing", "magnificent", "unrelenting"]}, {"metaphor": ["honesty", "brutal"], "baseline": ["genocidal", "referential", "atrocious"], "topic": "brutal honesty", "l1": ["fearless", "courageous", "despicable"]}, {"metaphor": ["idea", "bright"], "baseline": ["beautiful", "real", "supposed"], "topic": "bright idea", "l1": ["exciting", "wonderful", "intelligent"]}, {"metaphor": ["pride", "deflated"], "baseline": ["elated", "waning", "perverse"], "topic": "deflated pride", "l1": ["galvanized", "rejuvenated", "jubilant"]}, {"metaphor": ["shadows", "deepening"], "baseline": ["protracted", "watchful", "seductive"], "topic": "deepening shadows", "l1": ["unending", "enduring", "heightened"]}, {"metaphor": ["personality", "bubbly"], "baseline": ["personable", "narcissistic", "crisp"], "topic": "bubbly personality", "l1": ["endearing", "affable", "headstrong"]}, {"metaphor": ["poverty", "acute"], "baseline": ["socioeconomic", "symptomatic", "bipolar"], "topic": "acute poverty", "l1": ["prevalent", "worsening", "persistent"]}, {"metaphor": ["heat", "drowsy"], "baseline": ["relentless", "resistant", "helpless"], "topic": "drowsy heat", "l1": ["clammy", "drunk", "numb"]}, {"metaphor": ["breakdown", "nervous"], "baseline": ["serious", "skeptical", "angry"], "topic": "nervous breakdown", "l1": ["emotional", "dysfunctional", "mental"]}, {"metaphor": ["dunes", "graceful"], "baseline": ["slender", "expressive", "eloquent"], "topic": "graceful dunes", "l1": ["majestic", "fabled", "breathtaking"]}, {"metaphor": ["criticism", "heated"], "baseline": ["skeptical", "serious", "anti"], "topic": "heated criticism", "l1": ["furious", "partisan", "intense"]}, {"metaphor": ["breakdown", "mental"], "baseline": ["severe", "social", "significant"], "topic": "mental breakdown", "l1": ["marital", "organizational", "systemic"]}, {"metaphor": ["memories", "frozen"], "baseline": ["evocative", "fried", "awful"], "topic": "frozen memories", "l1": ["dormant", "fresh", "mad"]}, {"metaphor": ["city", "restless"], "baseline": ["agitated", "metropolitan", "demented"], "topic": "restless city", "l1": ["accustomed", "sleepless", "obsessed"]}, {"metaphor": ["personality", "colorful"], "baseline": ["narcissistic", "elegant", "arrogant"], "topic": "colorful personality", "l1": ["flamboyant", "playful", "humorous"]}, {"metaphor": ["emotions", "deflated"], "baseline": ["everyday", "elated", "stale"], "topic": "deflated emotions", "l1": ["overblown", "unrealistic", "galvanized"]}, {"metaphor": ["history", "dark"], "baseline": ["beautiful", "entire", "smoky"], "topic": "dark history", "l1": ["epic", "interesting", "medieval"]}, {"metaphor": ["statement", "strong"], "baseline": ["anti", "unacceptable", "similar"], "topic": "strong statement", "l1": ["renewed", "cautious", "recent"]}, {"metaphor": ["memories", "fleeting"], "baseline": ["enlightening", "terrifying", "flirtatious"], "topic": "fleeting memories", "l1": ["pleasurable", "heady", "lasting"]}, {"metaphor": ["desire", "burning"], "baseline": ["acrid", "willing", "selfish"], "topic": "burning desire", "l1": ["sacred", "holy", "angry"]}, {"metaphor": ["crisis", "deepening"], "baseline": ["transatlantic", "consequent", "devastating"], "topic": "deepening crisis", "l1": ["economic", "ongoing", "renewed"]}, {"metaphor": ["shadows", "ghostly"], "baseline": ["evil", "magical", "somber"], "topic": "ghostly shadows", "l1": ["shadowy", "faceless", "menacing"]}, {"metaphor": ["elite", "blind"], "baseline": ["amateur", "talented", "private"], "topic": "blind elite", "l1": ["incapable", "gifted", "ignorant"]}, {"metaphor": ["barrier", "cultural"], "baseline": ["ethnic", "dangerous", "archaeological"], "topic": "cultural barrier", "l1": ["psychological", "symbolic", "racial"]}, {"metaphor": ["stones", "quite"], "baseline": ["angry", "symbolic", "evident"], "topic": "quite stones", "l1": ["rare", "simple", "unknown"]}, {"metaphor": ["cold", "bitter"], "baseline": ["tough", "bittersweet", "common"], "topic": "bitter cold", "l1": ["harsh", "lingering", "prolonged"]}, {"metaphor": ["policy", "murky"], "baseline": ["puzzling", "awash", "chaotic"], "topic": "murky policy", "l1": ["contradictory", "vague", "unclear"]}, {"metaphor": ["dance", "delicate"], "baseline": ["elegant", "expressive", "beautiful"], "topic": "delicate dance", "l1": ["awkward", "graceful", "unusual"]}, {"metaphor": ["appearance", "ragged"], "baseline": ["regular", "final", "annual"], "topic": "ragged appearance", "l1": ["woeful", "miserable", "unkempt"]}, {"metaphor": ["remark", "colorful"], "baseline": ["elaborate", "elegant", "extravagant"], "topic": "colorful remark", "l1": ["humorous", "provocative", "memorable"]}, {"metaphor": ["face", "ancient"], "baseline": ["ancestral", "final", "pagan"], "topic": "ancient face", "l1": ["common", "familiar", "old"]}, {"metaphor": ["dress", "charming"], "baseline": ["plain", "amusing", "quaint"], "topic": "charming dress", "l1": ["silly", "elegant", "stylish"]}, {"metaphor": ["civilization", "hollow"], "baseline": ["devoid", "evil", "ancient"], "topic": "hollow civilization", "l1": ["lofty", "stony", "quaint"]}, {"metaphor": ["issue", "burning"], "baseline": ["angry", "bare", "smashing"], "topic": "burning issue", "l1": ["sacred", "incendiary", "holy"]}, {"metaphor": ["argument", "heated"], "baseline": ["searing", "tense", "brief"], "topic": "heated argument", "l1": ["partisan", "impassioned", "furious"]}, {"metaphor": ["poverty", "debilitating"], "baseline": ["socioeconomic", "circulatory", "rural"], "topic": "debilitating poverty", "l1": ["chronic", "acute", "catastrophic"]}, {"metaphor": ["desire", "unquenchable"], "baseline": ["ungrounded", "ravenous", "insatiable"], "topic": "unquenchable desire", "l1": ["newfound", "innate", "unwavering"]}, {"metaphor": ["darkness", "hollow"], "baseline": ["pleasant", "evil", "endless"], "topic": "hollow darkness", "l1": ["proverbial", "primitive", "devoid"]}, {"metaphor": ["desolation", "lonely"], "baseline": ["eternal", "dreamy", "insecure"], "topic": "lonely desolation", "l1": ["lonesome", "introspective", "forlorn"]}, {"metaphor": ["conversation", "empty"], "baseline": ["amusing", "quiet", "devoid"], "topic": "empty conversation", "l1": ["meaningless", "useless", "unguarded"]}, {"metaphor": ["morning", "crisp"], "baseline": ["shiny", "average", "spicy"], "topic": "crisp morning", "l1": ["brisk", "perfect", "nice"]}, {"metaphor": ["pride", "swelling"], "baseline": ["real", "religious", "acute"], "topic": "swelling pride", "l1": ["palpable", "dwindling", "immense"]}, {"metaphor": ["heights", "unforgiving"], "baseline": ["quiet", "lower", "airy"], "topic": "unforgiving heights", "l1": ["unimaginable", "breathtaking", "eternal"]}, {"metaphor": ["middle-class", "durable"], "baseline": ["cyclical", "thriving", "respectable"], "topic": "durable middle-class", "l1": ["desirable", "affordable", "decent"]}, {"metaphor": ["future", "bright"], "baseline": ["aware", "successful", "serious"], "topic": "bright future", "l1": ["bold", "glorious", "promising"]}, {"metaphor": ["house", "fiscal"], "baseline": ["annual", "sure", "regulatory"], "topic": "fiscal house", "l1": ["bipartisan", "federal", "legislative"]}, {"metaphor": ["city", "lost"], "baseline": ["alone", "catholic", "free"], "topic": "lost city", "l1": ["super", "unable", "real"]}, {"metaphor": ["future", "brighter"], "baseline": ["best", "supposed", "similar"], "topic": "brighter future", "l1": ["hopeful", "optimistic", "pessimistic"]}, {"metaphor": ["wage", "falling"], "baseline": ["sustained", "asleep", "vulnerable"], "topic": "falling wage", "l1": ["higher", "unchanged", "lower"]}, {"metaphor": ["class", "durable"], "baseline": ["speedy", "affluent", "competitive"], "topic": "durable class", "l1": ["affordable", "inclusive", "decent"]}, {"metaphor": ["greenery", "seductive"], "baseline": ["shimmering", "evocative", "vivacious"], "topic": "seductive greenery", "l1": ["enchanting", "otherworldly", "gorgeous"]}, {"metaphor": ["responsibility", "clear"], "baseline": ["alleged", "tough", "fiduciary"], "topic": "clear responsibility", "l1": ["aware", "wrong", "true"]}, {"metaphor": ["barrier", "financial"], "baseline": ["budgetary", "global", "superconducting"], "topic": "financial barrier", "l1": ["psychological", "vital", "mental"]}, {"metaphor": ["divide", "cultural"], "baseline": ["technological", "culinary", "archaeological"], "topic": "cultural divide", "l1": ["generational", "ideological", "racial"]}, {"metaphor": ["moment", "fleeting"], "baseline": ["emotional", "exciting", "final"], "topic": "fleeting moment", "l1": ["opportune", "teachable", "transformative"]}, {"metaphor": ["language", "flowery"], "baseline": ["formal", "shapely", "conversational"], "topic": "flowery language", "l1": ["vulgar", "descriptive", "racy"]}, {"metaphor": ["grave", "silent"], "baseline": ["faithful", "serial", "serious"], "topic": "silent grave", "l1": ["solemn", "somber", "poignant"]}, {"metaphor": ["area", "backward"], "baseline": ["authoritarian", "prior", "restricted"], "topic": "backward area", "l1": ["stationary", "primitive", "prone"]}, {"metaphor": ["anxiety", "crippling"], "baseline": ["sustained", "longstanding", "mental"], "topic": "crippling anxiety", "l1": ["impending", "worsening", "escalating"]}, {"metaphor": ["beaches", "lonely"], "baseline": ["restless", "plain", "solitary"], "topic": "lonely beaches", "l1": ["sleepy", "beautiful", "lovely"]}, {"metaphor": ["economy", "emerging"], "baseline": ["attractive", "worst", "untapped"], "topic": "emerging economy", "l1": ["global", "promising", "institutional"]}, {"metaphor": ["heights", "roaring"], "baseline": ["resounding", "ravenous", "strategic"], "topic": "roaring heights", "l1": ["giddy", "euphoric", "heady"]}, {"metaphor": ["collapse", "economic"], "baseline": ["corporate", "troubled", "budgetary"], "topic": "economic collapse", "l1": ["unlikely", "immediate", "renewed"]}, {"metaphor": ["solution", "clear-cut"], "baseline": ["consistent", "effective", "realistic"], "topic": "clear-cut solution", "l1": ["principled", "unturned", "defensible"]}, {"metaphor": ["ignorance", "acute"], "baseline": ["recurrent", "shameful", "superstitious"], "topic": "acute ignorance", "l1": ["alarming", "persistent", "mental"]}, {"metaphor": ["house", "charming"], "baseline": ["alone", "other", "playful"], "topic": "charming house", "l1": ["stately", "quiet", "good"]}, {"metaphor": ["answer", "clear-cut"], "baseline": ["supportable", "satisfied", "principled"], "topic": "clear-cut answer", "l1": ["predetermined", "succinct", "emphatic"]}, {"metaphor": ["hills", "rolling"], "baseline": ["sleepy", "alternating", "southern"], "topic": "rolling hills", "l1": ["random", "smashing", "idle"]}, {"metaphor": ["job", "green"], "baseline": ["bad", "sure", "easier"], "topic": "green job", "l1": ["brilliant", "nice", "different"]}, {"metaphor": ["competition", "cut-throat"], "baseline": ["streaky", "unscreened", "slovenly"], "topic": "cut-throat competition", "l1": ["undistorted", "faceless", "illusionary"]}, {"metaphor": ["statement", "clear"], "baseline": ["apparent", "unable", "preliminary"], "topic": "clear statement", "l1": ["satisfied", "aware", "unclear"]}, {"metaphor": ["baggage", "mental"], "baseline": ["spiritual", "oversized", "incoming"], "topic": "mental baggage", "l1": ["inadequate", "personal", "considerable"]}, {"metaphor": ["heart", "ancient"], "baseline": ["ailing", "different", "prehistoric"], "topic": "ancient heart", "l1": ["sacred", "holy", "historic"]}, {"metaphor": ["stones", "silent"], "baseline": ["ancient", "holy", "stoic"], "topic": "silent stones", "l1": ["unreleased", "anonymous", "unknown"]}, {"metaphor": ["class", "healthy"], "baseline": ["affluent", "supportive", "ready"], "topic": "healthy class", "l1": ["competitive", "comfortable", "normal"]}, {"metaphor": ["ego", "filthy"], "baseline": ["grimy", "unclean", "stinking"], "topic": "filthy ego", "l1": ["childish", "stupid", "unhealthy"]}, {"metaphor": ["criticism", "heavy"], "baseline": ["sporadic", "anti", "skeptical"], "topic": "heavy criticism", "l1": ["intense", "excessive", "occasional"]}, {"metaphor": ["cold", "biting"], "baseline": ["terrible", "bad", "hardened"], "topic": "biting cold", "l1": ["blistering", "vicious", "merciless"]}, {"metaphor": ["competition", "fierce"], "baseline": ["serious", "angry", "bruising"], "topic": "fierce competition", "l1": ["spirited", "formidable", "frenzied"]}, {"metaphor": ["heat", "debilitating"], "baseline": ["rheumatoid", "serious", "intestinal"], "topic": "debilitating heat", "l1": ["unbearable", "mental", "immune"]}, {"metaphor": ["beauty", "raw"], "baseline": ["agricultural", "enchanting", "everyday"], "topic": "raw beauty", "l1": ["precious", "pure", "abundant"]}, {"metaphor": ["ego", "big"], "baseline": ["archetypal", "sizable", "great"], "topic": "big ego", "l1": ["mega", "tremendous", "ultimate"]}, {"metaphor": ["expenditure", "ballooning"], "baseline": ["recurrent", "shrunken", "insufficient"], "topic": "ballooning expenditure", "l1": ["unsustainable", "astronomic", "voluminous"]}, {"metaphor": ["productivity", "falling"], "baseline": ["attributable", "stimulating", "unexpected"], "topic": "falling productivity", "l1": ["higher", "advancing", "average"]}, {"metaphor": ["hands", "silky"], "baseline": ["bad", "spiky", "supposed"], "topic": "silky hands", "l1": ["comfortable", "easy", "graceful"]}, {"metaphor": ["engine", "roaring"], "baseline": ["auxiliary", "rippling", "supersonic"], "topic": "roaring engine", "l1": ["mighty", "muffled", "thirsty"]}, {"metaphor": ["solution", "elegant"], "baseline": ["delightful", "appropriate", "pragmatic"], "topic": "elegant solution", "l1": ["simple", "sensible", "easy"]}, {"metaphor": ["character", "filthy"], "baseline": ["fictional", "appalling", "familiar"], "topic": "filthy character", "l1": ["stupid", "nasty", "vulgar"]}, {"metaphor": ["economy", "growing"], "baseline": ["anemic", "worried", "steady"], "topic": "growing economy", "l1": ["global", "thriving", "recent"]}, {"metaphor": ["weather", "brutal"], "baseline": ["sadistic", "barbaric", "economic"], "topic": "brutal weather", "l1": ["worst", "atrocious", "terrible"]}, {"metaphor": ["concentration", "fierce"], "baseline": ["renewed", "allowable", "specific"], "topic": "fierce concentration", "l1": ["allied", "partisan", "notorious"]}, {"metaphor": ["ego", "bloated"], "baseline": ["unrecognizable", "psychic", "stagnant"], "topic": "bloated ego", "l1": ["overblown", "colossal", "unaccountable"]}, {"metaphor": ["cost", "high"], "baseline": ["additional", "concerned", "total"], "topic": "high cost", "l1": ["lower", "new", "average"]}, {"metaphor": ["cost", "political"], "baseline": ["additional", "considerable", "profitable"], "topic": "political cost", "l1": ["personal", "internal", "financial"]}, {"metaphor": ["empire", "lost"], "baseline": ["fledgling", "allied", "celestial"], "topic": "lost empire", "l1": ["valuable", "victorious", "super"]}, {"metaphor": ["afternoon", "golden"], "baseline": ["recent", "bald", "somber"], "topic": "golden afternoon", "l1": ["inaugural", "royal", "outstanding"]}, {"metaphor": ["anxiety", "social"], "baseline": ["significant", "religious", "basic"], "topic": "social anxiety", "l1": ["underlying", "psychological", "profound"]}, {"metaphor": ["faith", "deep"], "baseline": ["bitter", "true", "subtle"], "topic": "deep faith", "l1": ["abiding", "sincere", "enduring"]}, {"metaphor": ["expenditure", "vast"], "baseline": ["rural", "revised", "sprawling"], "topic": "vast expenditure", "l1": ["sizeable", "substantial", "devoted"]}, {"metaphor": ["ignorance", "blissful"], "baseline": ["superstitious", "societal", "wishful"], "topic": "blissful ignorance", "l1": ["momentary", "smug", "paradoxical"]}, {"metaphor": ["language", "soft"], "baseline": ["dry", "archaic", "pliable"], "topic": "soft language", "l1": ["easy", "delicate", "familiar"]}, {"metaphor": ["darkness", "consuming"], "baseline": ["somber", "intensive", "palpable"], "topic": "consuming darkness", "l1": ["endless", "intense", "dangerous"]}, {"metaphor": ["eyes", "cold"], "baseline": ["arctic", "severe", "rainy"], "topic": "cold eyes", "l1": ["hungry", "terrible", "frozen"]}, {"metaphor": ["demand", "expanding"], "baseline": ["worsening", "ambitious", "underlying"], "topic": "expanding demand", "l1": ["mutual", "encouraging", "bilateral"]}, {"metaphor": ["hills", "graceful"], "baseline": ["sleek", "sensuous", "splendid"], "topic": "graceful hills", "l1": ["majestic", "beautiful", "spectacular"]}, {"metaphor": ["place", "unforgiving"], "baseline": ["appropriate", "final", "implacable"], "topic": "unforgiving place", "l1": ["cruel", "perilous", "grueling"]}, {"metaphor": ["frontier", "raw"], "baseline": ["ancient", "turbulent", "autonomous"], "topic": "raw frontier", "l1": ["valuable", "wild", "seasoned"]}, {"metaphor": ["demand", "growing"], "baseline": ["alarming", "weak", "waning"], "topic": "growing demand", "l1": ["global", "recent", "concerned"]}, {"metaphor": ["face", "clouded"], "baseline": ["ready", "inconclusive", "inevitable"], "topic": "clouded face", "l1": ["worsening", "thorny", "uncertain"]}, {"metaphor": ["appearance", "cold"], "baseline": ["annual", "slow", "severe"], "topic": "cold appearance", "l1": ["usual", "strange", "recent"]}, {"metaphor": ["loneliness", "cosmic"], "baseline": ["extraterrestrial", "sensuous", "existential"], "topic": "cosmic loneliness", "l1": ["eternal", "metaphysical", "infinite"]}, {"metaphor": ["honesty", "unwavering"], "baseline": ["truthful", "innate", "moral"], "topic": "unwavering honesty", "l1": ["admirable", "exemplary", "fearless"]}, {"metaphor": ["hands", "velvet"], "baseline": ["silvery", "bad", "gray"], "topic": "velvet hands", "l1": ["orange", "yellow", "padded"]}, {"metaphor": ["weakness", "big"], "baseline": ["apparent", "systemic", "sluggish"], "topic": "big weakness", "l1": ["corporate", "tremendous", "bad"]}]

// var conds = {0:{"literal":["warm"],"metaphorical":["intense"]}}

var maxes = [120,120,120,120,120,120,120,120,120,120,120,120]

var random_indices = maxes.map(x => getRandomInt(x))

// _.map(getRandomInt,maxes)
// const map1 = array1.map(x => x * 2);

// var foo = getRandomInt(120)

var items = [
	
	[
	data[random_indices[0]]["l1"][0],
	data[random_indices[0]]["baseline"][0],
	data[random_indices[0]]["l1"][1],
	data[random_indices[0]]["baseline"][1],
	],

	[
	data[random_indices[1]]["l1"][0],
	data[random_indices[1]]["baseline"][0],
	data[random_indices[1]]["l1"][1],
	data[random_indices[1]]["baseline"][1],
	],

	[
	data[random_indices[2]]["l1"][0],
	data[random_indices[2]]["baseline"][0],
	data[random_indices[2]]["l1"][1],
	data[random_indices[2]]["baseline"][1],
	],

	[
	data[random_indices[3]]["l1"][0],
	data[random_indices[3]]["baseline"][0],
	data[random_indices[3]]["l1"][1],
	data[random_indices[3]]["baseline"][1],
	],

	[
	data[random_indices[4]]["l1"][0],
	data[random_indices[4]]["baseline"][0],
	data[random_indices[4]]["l1"][1],
	data[random_indices[4]]["baseline"][1],
	],

	[
	data[random_indices[5]]["l1"][0],
	data[random_indices[5]]["baseline"][0],
	data[random_indices[5]]["l1"][1],
	data[random_indices[5]]["baseline"][1],
	],

	[
	data[random_indices[6]]["l1"][0],
	data[random_indices[6]]["baseline"][0],
	data[random_indices[6]]["l1"][1],
	data[random_indices[6]]["baseline"][1],
	],

	[
	data[random_indices[7]]["l1"][0],
	data[random_indices[7]]["baseline"][0],
	data[random_indices[7]]["l1"][1],
	data[random_indices[7]]["baseline"][1],
	],

	[
	data[random_indices[8]]["l1"][0],
	data[random_indices[8]]["baseline"][0],
	data[random_indices[8]]["l1"][1],
	data[random_indices[8]]["baseline"][1],
	],

	[
	data[random_indices[9]]["l1"][0],
	data[random_indices[9]]["baseline"][0],
	data[random_indices[9]]["l1"][1],
	data[random_indices[9]]["baseline"][1],
	],

	[
	data[random_indices[10]]["l1"][0],
	data[random_indices[10]]["baseline"][0],
	data[random_indices[10]]["l1"][1],
	data[random_indices[10]]["baseline"][1],
	],

	[
	data[random_indices[11]]["l1"][0],
	data[random_indices[11]]["baseline"][0],
	data[random_indices[11]]["l1"][1],
	data[random_indices[11]]["baseline"][1],
	],

	]

var topics = [
	data[random_indices[0]]["topic"], 
	data[random_indices[1]]["topic"],
	data[random_indices[2]]["topic"],
	data[random_indices[3]]["topic"],
	data[random_indices[4]]["topic"],
	data[random_indices[5]]["topic"], 
	data[random_indices[6]]["topic"],
	data[random_indices[7]]["topic"],
	data[random_indices[8]]["topic"],
	data[random_indices[9]]["topic"],
	data[random_indices[10]]["topic"],
	data[random_indices[11]]["topic"],
	]

// var items = shuffle(items)
// ten conditions
// dictionary from condition to list of 12 baseline and list of 12 system

var condition=0




var exp_task_description = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: 
		"<h3>Instructions</h3><p>In this task, you'll be shown expressions, like \"colorful personality\" or \"fiery temper\". Your goal is to help us understand what they mean. For example, a colorful personality might be a personality that is eccentric or fun.</p>"
	
    	// "<p>We'll show you an expression like \"colorful personality\" and ask you to rate how good different words (like \"eccentric\" or \"fun\") are at explaining what the noun (e.g. \"personality\") is like.</p>"
    	+
		"<p>Press \"(q)\" to see an example. </p>"
};


// var example = {
//     type: 'html-button-response',
//     stimulus: "<h3>Example: <b>fiery temper</b></h3>"
//     	+
//   		'<p>Here is an example. "intense" is preferable to "warm", because a fiery temper is warm but not intense. Select "intense" to continue.</p>',
//     choices: ["intense","warm"],
//     prompt: "<br><br>"
// 	}

var example = {
    type: 'html-multi-slider-response',
    stimulus: "<h3>Example: <b>fiery temper</b></h3>"
    	+
  		'<p>Here is an example. A fiery temper is intense but not warm. You should rate "intense" as highly relevant, and "warm" as highly irrelevant.',

  	
  	labels: ["not relevant","relevant"],
  	prompt: ["intense","warm"],
     //choices: shuffle([conds[condition]["literal"][0],conds[condition]["metaphorical"][0]]),
     //prompt: "<br>Which adjective is better?<br>"
	}

// var exp0 = {
//     type: 'survey-multi-choice',
//     questions: [
//     	{prompt: 'Sentence:'+topics[0], options: [items[0],items[0]], required:true,},
//     	{prompt: 'Sentence:'+topics[0], options: [items[0],items[0]], required:true,}, 
//     	]
// 	}
//var i = 0
//var exp0 = {
 //    type: 'html-multi-slider-response',
 //    stimulus:
 //  	'Sentence:'+topics[i],
 //  	labels: shuffle([items[i][0],items[i][1]]),
// 	}

var exp = maxes

// items.length

var i;
for (i = 0; i < items.length; i++) { 
exp[i] = {
    type: 'html-multi-slider-response',
    stimulus:
  	'Sentence: <b>'+topics[i]+'</b>',
  	labels: ["not relevant","relevant"],
  	prompt: shuffle([items[i][0],items[i][1],items[i][2],items[i][3]]),
	};
}

var end_exp = {
	type: "html-keyboard-response",
	stimulus: "<p id='bounded'>Thank you for participating!<br\><br\>Press any key to end the survey.</p>",
	on_load: function() {
		spacebar_active = false;
	}};

/******************************
 *	Timeline composition
 ******************************/



timeline.push(exp_task_description);
timeline.push(example);
timeline.push(exp[0]);
timeline.push(exp[1]);
timeline.push(exp[2]);
timeline.push(exp[3]);
timeline.push(exp[4]);
timeline.push(exp[5]);
timeline.push(exp[6]);
timeline.push(exp[7]);
timeline.push(exp[8]);
timeline.push(exp[9]);
timeline.push(exp[10]);
timeline.push(exp[11]);
timeline.push(end_exp);


/* preload images */
var image_paths = [];
for (var i = stimuli.length - 1; i >= 0; i--) {
	image_paths.push(stimuli[i].stimulus);
}

/* start the experiment */
jsPsych.init({
	preload_images: image_paths,
	timeline: timeline,
	on_finish: function() {
		var saveAllData = function(callback) {

			if (USING_PSITURK) {
				psiturk.taskdata.set('bonus', 0);
			}

			//save to psiturk database
			callback();
		};

		saveAllData(function() {
			if (USING_PSITURK) {
				console.log('Saving data to database and ending experiment.');
				psiturk.saveData({
					success: function(){
						psiturk.computeBonus('compute_bonus', function() {
							psiturk.completeHIT();
						});
					}
				});
			} else {
				jsPsych.data.displayData();
				var jsPsychData = JSON.parse(jsPsych.data.get().json());
				// downloadJson(jsPsychData, 'data.json');
			}
		});
	},
	on_data_update: function(data) {
			if (USING_PSITURK) { psiturk.recordTrialData(data); }
	}
});
