
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

var data = [{"l1": ["ideological", "fiscal", "good"], "baseline": ["philosophical", "administrative", "greater"], "topic": "academic gap", "metaphor": ["gap", "academic"]}, {"l1": ["widespread", "alarming", "extreme"], "baseline": ["abject", "escalating", "reckless"], "topic": "acute ignorance", "metaphor": ["ignorance", "acute"]}, {"l1": ["worsening", "prevalent", "persistent"], "baseline": ["adverse", "global", "ethnic"], "topic": "acute poverty", "metaphor": ["poverty", "acute"]}, {"l1": ["timeless", "immortal", "unchanging"], "baseline": ["rejuvenated", "mellow", "unfathomable"], "topic": "ageless rhythms", "metaphor": ["rhythms", "ageless"]}, {"l1": ["ambitious", "innovative", "effective"], "baseline": ["outspoken", "risky", "academic"], "topic": "aggressive program", "metaphor": ["program", "aggressive"]}, {"l1": ["unproductive", "crummy", "stupendous"], "baseline": ["ungodly", "undoable", "mundane"], "topic": "backbreaking rent", "metaphor": ["rent", "backbreaking"]}, {"l1": ["impoverished", "prone", "poor"], "baseline": ["parallel", "important", "thrusting"], "topic": "backward area", "metaphor": ["area", "backward"]}, {"l1": ["feudal", "authoritarian", "primitive"], "baseline": ["great", "aristocratic", "founding"], "topic": "backward tradition", "metaphor": ["tradition", "backward"]}, {"l1": ["optimal", "budgetary", "vigorous"], "baseline": ["potential", "adequate", "flexible"], "topic": "balanced growth", "metaphor": ["growth", "balanced"]}, {"l1": ["unsustainable", "voluminous", "frugal"], "baseline": ["macroeconomic", "insufficient", "substantial"], "topic": "ballooning expenditure", "metaphor": ["expenditure", "ballooning"]}, {"l1": ["potential", "tremendous", "surprising"], "baseline": ["ready", "mild", "new"], "topic": "big weakness", "metaphor": ["weakness", "big"]}, {"l1": ["amusing", "cynical", "clever"], "baseline": ["nasty", "afraid", "worried"], "topic": "biting look", "metaphor": ["look", "biting"]}, {"l1": ["unrelenting", "magnificent", "majestic"], "baseline": ["windy", "torrid", "melancholic"], "topic": "blazing desolation", "metaphor": ["desolation", "blazing"]}, {"l1": ["gifted", "renowned", "younger"], "baseline": ["illiterate", "national", "affluent"], "topic": "blind elite", "metaphor": ["elite", "blind"]}, {"l1": ["evil", "suicidal", "dumb"], "baseline": ["angry", "nasty", "eyed"], "topic": "blind hate", "metaphor": ["hate", "blind"]}, {"l1": ["naive", "suicidal", "reckless"], "baseline": ["poor", "bald", "modest"], "topic": "blind optimism", "metaphor": ["optimism", "blind"]}, {"l1": ["wasteful", "unsustainable", "excess"], "baseline": ["lifeless", "tough", "modest"], "topic": "bloated spending", "metaphor": ["spending", "bloated"]}, {"l1": ["thriving", "vibrant", "ripe"], "baseline": ["pigtailed", "unraveled", "domestic"], "topic": "blossoming industry", "metaphor": ["industry", "blossoming"]}, {"l1": ["beautiful", "particular", "true"], "baseline": ["fearful", "sincere", "unrequited"], "topic": "blue feelings", "metaphor": ["feelings", "blue"]}, {"l1": ["quiet", "nondescript", "distant"], "baseline": ["forlorn", "patterned", "little"], "topic": "blue obscurity", "metaphor": ["obscurity", "blue"]}, {"l1": ["refreshing", "copious", "fresh"], "baseline": ["unopened", "divine", "artistic"], "topic": "bottled passion", "metaphor": ["passion", "bottled"]}, {"l1": ["unchecked", "continual", "unbridled"], "baseline": ["phenomenal", "zippy", "smoother"], "topic": "breakneck expansion", "metaphor": ["expansion", "breakneck"]}, {"l1": ["unable", "wrong", "serious"], "baseline": ["heavy", "anxious", "possible"], "topic": "broken hope", "metaphor": ["hope", "broken"]}, {"l1": ["minor", "whole", "similar"], "baseline": ["good", "numerous", "graceful"], "topic": "broken melody", "metaphor": ["melody", "broken"]}, {"l1": ["subsidized", "costly", "bankrupt"], "baseline": ["senior", "insolvent", "voluntary"], "topic": "burdened service", "metaphor": ["service", "burdened"]}, {"l1": ["sacred", "holy", "countless"], "baseline": ["yellow", "kindred", "crazy"], "topic": "burning soul", "metaphor": ["soul", "burning"]}, {"l1": ["disoriented", "contorted", "gaping"], "baseline": ["unexplored", "undulating", "overcrowded"], "topic": "choked gullies", "metaphor": ["gullies", "choked"]}, {"l1": ["rampant", "worsening", "prevalent"], "baseline": ["acute", "fatal", "humanitarian"], "topic": "chronic poverty", "metaphor": ["poverty", "chronic"]}, {"l1": ["moral", "secular", "sporting"], "baseline": ["parliamentary", "flexible", "artistic"], "topic": "civic fabric", "metaphor": ["fabric", "civic"]}, {"l1": ["aware", "direct", "immediate"], "baseline": ["administrative", "unable", "impossible"], "topic": "clear responsibility", "metaphor": ["responsibility", "clear"]}, {"l1": ["succinct", "emphatic", "conclusive"], "baseline": ["goddamned", "possible", "obvious"], "topic": "clear-cut answer", "metaphor": ["answer", "clear-cut"]}, {"l1": ["principled", "definite", "conclusive"], "baseline": ["lasting", "flexible", "orthographic"], "topic": "clear-cut solution", "metaphor": ["solution", "clear-cut"]}, {"l1": ["pessimistic", "bleak", "mindful"], "baseline": ["worried", "serious", "simmering"], "topic": "clouded future", "metaphor": ["future", "clouded"]}, {"l1": ["foreseeable", "pessimistic", "bleak"], "baseline": ["windless", "worrisome", "ominous"], "topic": "cloudy prospect", "metaphor": ["prospect", "cloudy"]}, {"l1": ["usual", "strange", "recent"], "baseline": ["wet", "unlikely", "mild"], "topic": "cold appearance", "metaphor": ["appearance", "cold"]}, {"l1": ["wrong", "new", "possible"], "baseline": ["warmer", "appropriate", "true"], "topic": "cold justice", "metaphor": ["justice", "cold"]}, {"l1": ["precarious", "creaking", "catastrophic"], "baseline": ["adequate", "prolonged", "eroding"], "topic": "collapsing health", "metaphor": ["health", "collapsing"]}, {"l1": ["accidental", "divergent", "disagreeable"], "baseline": ["unspoken", "intolerable", "irrefutable"], "topic": "colliding contradiction", "metaphor": ["contradiction", "colliding"]}, {"l1": ["flamboyant", "humorous", "playful"], "baseline": ["multicolored", "garish", "entertaining"], "topic": "colorful personality", "metaphor": ["personality", "colorful"]}, {"l1": ["provocative", "unflattering", "humorous"], "baseline": ["memorable", "charming", "exuberant"], "topic": "colorful remark", "metaphor": ["remark", "colorful"]}, {"l1": ["optimistic", "generous", "realistic"], "baseline": ["national", "ambitious", "virtuous"], "topic": "compassionate budget", "metaphor": ["budget", "compassionate"]}, {"l1": ["productive", "scarce", "vast"], "baseline": ["active", "hydrochloric", "needy"], "topic": "concentrated poverty", "metaphor": ["poverty", "concentrated"]}, {"l1": ["endless", "intense", "alone"], "baseline": ["dreamy", "unrelenting", "voracious"], "topic": "consuming darkness", "metaphor": ["darkness", "consuming"]}, {"l1": ["traditional", "fresh", "easy"], "baseline": ["homemade", "organizational", "pleasing"], "topic": "corporate pie", "metaphor": ["pie", "corporate"]}, {"l1": ["pervasive", "debilitating", "oppressive"], "baseline": ["addictive", "corporate", "flammable"], "topic": "corrosive corruption", "metaphor": ["corruption", "corrosive"]}, {"l1": ["dangerous", "possible", "wrong"], "baseline": ["alleged", "guilty", "logical"], "topic": "criminal path", "metaphor": ["path", "criminal"]}, {"l1": ["unbearable", "nagging", "inevitable"], "baseline": ["escalating", "playful", "deteriorating"], "topic": "crippling awkwardness", "metaphor": ["awkwardness", "crippling"]}, {"l1": ["awash", "treacherous", "countless"], "baseline": ["undulating", "unprecedented", "nightmarish"], "topic": "crisscrossed chaos", "metaphor": ["chaos", "crisscrossed"]}, {"l1": ["relentless", "debilitating", "worst"], "baseline": ["battered", "periodic", "normal"], "topic": "crushing cycle", "metaphor": ["cycle", "crushing"]}, {"l1": ["conquering", "immense", "convincing"], "baseline": ["embarrassing", "overwhelming", "actual"], "topic": "crushing difficulty", "metaphor": ["difficulty", "crushing"]}, {"l1": ["disastrous", "debilitating", "convincing"], "baseline": ["fierce", "avenging", "overwhelming"], "topic": "crushing effect", "metaphor": ["effect", "crushing"]}, {"l1": ["crippling", "debilitating", "miserable"], "baseline": ["afflicted", "nationwide", "heartbreaking"], "topic": "crushing hunger", "metaphor": ["hunger", "crushing"]}, {"l1": ["terrible", "oppressive", "miserable"], "baseline": ["absolute", "innate", "heartbreaking"], "topic": "crushing ignorance", "metaphor": ["ignorance", "crushing"]}, {"l1": ["conquering", "heartbreaking", "immense"], "baseline": ["appalling", "gigantic", "worsening"], "topic": "crushing misery", "metaphor": ["misery", "crushing"]}, {"l1": ["successive", "heartbreaking", "debilitating"], "baseline": ["aggravated", "gigantic", "ruthless"], "topic": "crushing neglect", "metaphor": ["neglect", "crushing"]}, {"l1": ["successive", "miserable", "catastrophic"], "baseline": ["fiscal", "brutal", "prolonged"], "topic": "crushing unemployment", "metaphor": ["unemployment", "crushing"]}, {"l1": ["inherent", "legitimate", "conducive"], "baseline": ["ancient", "glaring", "societal"], "topic": "cultural impediment", "metaphor": ["impediment", "cultural"]}, {"l1": ["enduring", "emotional", "psychological"], "baseline": ["aesthetic", "newfound", "geographic"], "topic": "cultural strength", "metaphor": ["strength", "cultural"]}, {"l1": ["undistorted", "faceless", "spiteful"], "baseline": ["wolfish", "pitiless", "individualistic"], "topic": "cut-throat competition", "metaphor": ["competition", "cut-throat"]}, {"l1": ["romantic", "pure", "quiet"], "baseline": ["youthful", "pale", "white"], "topic": "dark passion", "metaphor": ["passion", "dark"]}, {"l1": ["alone", "unclear", "afraid"], "baseline": ["male", "additional", "sure"], "topic": "dead money", "metaphor": ["money", "dead"]}, {"l1": ["unbearable", "mental", "chronic"], "baseline": ["afflicted", "worse", "financial"], "topic": "debilitating poverty", "metaphor": ["poverty", "debilitating"]}, {"l1": ["difficult", "obvious", "simple"], "baseline": ["preliminary", "analytical", "fresh"], "topic": "deep analysis", "metaphor": ["analysis", "deep"]}, {"l1": ["enduring", "evident", "widespread"], "baseline": ["homogeneous", "big", "prevailing"], "topic": "deep inequality", "metaphor": ["inequality", "deep"]}, {"l1": ["enduring", "rich", "inner"], "baseline": ["afflicted", "global", "underprivileged"], "topic": "deep poverty", "metaphor": ["poverty", "deep"]}, {"l1": ["certain", "other", "lower"], "baseline": ["painful", "huge", "senior"], "topic": "deep rank", "metaphor": ["rank", "deep"]}, {"l1": ["spiritual", "immense", "enduring"], "baseline": ["otherworldly", "heavy", "tremendous"], "topic": "deep solitude", "metaphor": ["solitude", "deep"]}, {"l1": ["longstanding", "unshakeable", "dualistic"], "baseline": ["baneful", "underlying", "evil"], "topic": "deep-rooted belief", "metaphor": ["belief", "deep-rooted"]}, {"l1": ["longstanding", "generational", "societal"], "baseline": ["ingrown", "hallowed", "old"], "topic": "deep-rooted tradition", "metaphor": ["tradition", "deep-rooted"]}, {"l1": ["economic", "ongoing", "renewed"], "baseline": ["bilateral", "unprecedented", "multilateral"], "topic": "deepening crisis", "metaphor": ["crisis", "deepening"]}, {"l1": ["greater", "moral", "underlying"], "baseline": ["alarming", "impoverished", "warmer"], "topic": "deeper poverty", "metaphor": ["poverty", "deeper"]}, {"l1": ["euphoric", "giddy", "overblown"], "baseline": ["despondent", "chagrined", "unruffled"], "topic": "deflated emotions", "metaphor": ["emotions", "deflated"]}, {"l1": ["overblown", "depressing", "queasy"], "baseline": ["bizarre", "depressed", "lighthearted"], "topic": "deflated joke", "metaphor": ["joke", "deflated"]}, {"l1": ["stale", "unrealistic", "waning"], "baseline": ["wobbly", "metaphorical", "certain"], "topic": "deflated meaning", "metaphor": ["meaning", "deflated"]}, {"l1": ["jubilant", "galvanized", "waning"], "baseline": ["buoyant", "wobbly", "tremendous"], "topic": "deflated pride", "metaphor": ["pride", "deflated"]}, {"l1": ["appalling", "wretched", "oppressive"], "baseline": ["alarming", "inhuman", "unromantic"], "topic": "dehumanizing poverty", "metaphor": ["poverty", "dehumanizing"]}, {"l1": ["bittersweet", "unimaginable", "unbelievable"], "baseline": ["salty", "baked", "indescribable"], "topic": "delicious agony", "metaphor": ["agony", "delicious"]}, {"l1": ["cultural", "genuine", "greater"], "baseline": ["astonishing", "parliamentary", "incumbent"], "topic": "democratic vitality", "metaphor": ["vitality", "democratic"]}, {"l1": ["quiet", "idyllic", "lonely"], "baseline": ["important", "everlasting", "secluded"], "topic": "deserted friendship", "metaphor": ["friendship", "deserted"]}, {"l1": ["lonely", "idyllic", "unnoticed"], "baseline": ["monogamous", "featureless", "enduring"], "topic": "deserted relationships", "metaphor": ["relationships", "deserted"]}, {"l1": ["serene", "otherworldly", "pristine"], "baseline": ["overgrown", "lyrical", "unforgiving"], "topic": "desolate beauty", "metaphor": ["beauty", "desolate"]}, {"l1": ["bleak", "glum", "precarious"], "baseline": ["ghastly", "enduring", "unfortunate"], "topic": "dim reminder", "metaphor": ["reminder", "dim"]}, {"l1": ["pessimistic", "bleak", "mulled"], "baseline": ["enticing", "heightened", "possible"], "topic": "dimmed prospect", "metaphor": ["prospect", "dimmed"]}, {"l1": ["wicked", "dishonest", "terrible"], "baseline": ["virtuous", "righteous", "nasty"], "topic": "dirty deeds", "metaphor": ["deeds", "dirty"]}, {"l1": ["cynical", "vulgar", "childish"], "baseline": ["lustful", "insatiable", "altruistic"], "topic": "dirty desires", "metaphor": ["desires", "dirty"]}, {"l1": ["supreme", "founding", "turbulent"], "baseline": ["constitutional", "socialist", "organic"], "topic": "dissolved time", "metaphor": ["time", "dissolved"]}, {"l1": ["sleazy", "seedy", "rowdy"], "baseline": ["new", "formal", "untrustworthy"], "topic": "dodgy bar", "metaphor": ["bar", "dodgy"]}, {"l1": ["rewarding", "excess", "vast"], "baseline": ["dry", "lavish", "potential"], "topic": "draining expense", "metaphor": ["expense", "draining"]}, {"l1": ["good", "higher", "affordable"], "baseline": ["poor", "efficient", "educated"], "topic": "durable class", "metaphor": ["class", "durable"]}, {"l1": ["desirable", "modest", "decent"], "baseline": ["residential", "fragile", "comprehensive"], "topic": "durable middle-class", "metaphor": ["middle-class", "durable"]}, {"l1": ["interactive", "competitive", "creative"], "baseline": ["nonlinear", "real", "consolidated"], "topic": "dynamic company", "metaphor": ["company", "dynamic"]}, {"l1": ["intelligent", "creative", "energetic"], "baseline": ["narcissistic", "aggressive", "likable"], "topic": "dynamic personality", "metaphor": ["personality", "dynamic"]}, {"l1": ["consequent", "structural", "fragile"], "baseline": ["apparent", "freshwater", "sustainable"], "topic": "ecological collapse", "metaphor": ["collapse", "ecological"]}, {"l1": ["civil", "disastrous", "tough"], "baseline": ["sluggish", "allied", "pivotal"], "topic": "economic battle", "metaphor": ["battle", "economic"]}, {"l1": ["immediate", "immense", "imminent"], "baseline": ["atomic", "illegal", "regional"], "topic": "economic destruction", "metaphor": ["destruction", "economic"]}, {"l1": ["scientific", "theoretical", "international"], "baseline": ["potential", "advanced", "sluggish"], "topic": "economic field", "metaphor": ["field", "economic"]}, {"l1": ["federal", "necessary", "civil"], "baseline": ["southern", "aggressive", "potential"], "topic": "economic force", "metaphor": ["force", "economic"]}, {"l1": ["stellar", "untold", "excess"], "baseline": ["humiliating", "regional", "awful"], "topic": "economic heap", "metaphor": ["heap", "economic"]}, {"l1": ["cyclical", "breakneck", "tepid"], "baseline": ["zippy", "sluggish", "unprecedented"], "topic": "economic laggard", "metaphor": ["laggard", "economic"]}, {"l1": ["academic", "practical", "basic"], "baseline": ["advanced", "recent", "other"], "topic": "economic medicine", "metaphor": ["medicine", "economic"]}, {"l1": ["imminent", "speculative", "impending"], "baseline": ["recessionary", "substantial", "agricultural"], "topic": "economic meltdown", "metaphor": ["meltdown", "economic"]}, {"l1": ["entrepreneurial", "greater", "organizational"], "baseline": ["agile", "possible", "social"], "topic": "economic mobility", "metaphor": ["mobility", "economic"]}, {"l1": ["internal", "sustained", "lower"], "baseline": ["neurological", "apparent", "robust"], "topic": "economic muscle", "metaphor": ["muscle", "economic"]}, {"l1": ["easy", "traditional", "fresh"], "baseline": ["possible", "social", "wonderful"], "topic": "economic pie", "metaphor": ["pie", "economic"]}, {"l1": ["bipartisan", "anti", "appropriate"], "baseline": ["political", "affordable", "worsening"], "topic": "economic prescription", "metaphor": ["prescription", "economic"]}, {"l1": ["ambitious", "vibrant", "civic"], "baseline": ["strategic", "postwar", "harmonious"], "topic": "economic revitalization", "metaphor": ["revitalization", "economic"]}, {"l1": ["recent", "likely", "quarterly"], "baseline": ["slight", "annual", "latest"], "topic": "economic rise", "metaphor": ["rise", "economic"]}, {"l1": ["dramatic", "depressing", "disastrous"], "baseline": ["recent", "greater", "alarming"], "topic": "economic slide", "metaphor": ["slide", "economic"]}, {"l1": ["immense", "intellectual", "geographical"], "baseline": ["overall", "analogous", "global"], "topic": "economic sphere", "metaphor": ["sphere", "economic"]}, {"l1": ["vibrant", "medieval", "multicultural"], "baseline": ["worse", "grotesque", "airy"], "topic": "economic tapestry", "metaphor": ["tapestry", "economic"]}, {"l1": ["important", "vital", "practical"], "baseline": ["potential", "automated", "greater"], "topic": "economic tool", "metaphor": ["tool", "economic"]}, {"l1": ["racial", "mutual", "ideological"], "baseline": ["relevant", "alone", "obvious"], "topic": "educational gap", "metaphor": ["gap", "educational"]}, {"l1": ["basic", "dynamic", "underlying"], "baseline": ["arbitrary", "graphical", "nonlinear"], "topic": "embedded inequality", "metaphor": ["inequality", "embedded"]}]
// var conds = {0:{"literal":["warm"],"metaphorical":["intense"]}}

var max_num = 110

var maxes = [max_num,max_num,max_num,max_num,max_num,max_num,max_num,max_num,max_num,max_num,max_num,max_num]

var random_indices = []

// maxes.map(x => getRandomInt(x))

for (i = 0; i < 100; i++) {
  if (random_indices.length == 12) { break; }
  random_index = getRandomInt(max_num)
  if (!random_indices.includes(random_index)) { random_indices.push(random_index); }
}

// _.map(getRandomInt,maxes)
// const map1 = array1.map(x => x * 2);

// var foo = getRandomInt(max_num)

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
  		// '<p>Here is an example. A fiery temper is intense but not warm or cold. You should rate "intense" as highly relevant, and "warm" and "cold" as highly irrelevant.',

  		'<p>Here is an example. Rate "intense" as <i>relevant</i>, because a fiery temper is an intense temper.</p>'+
  		'<p>Rate "calm" as <i>relevant</i> too, because a fiery temper is <b>not</b> calm.</p>'+
  		'<p>Rate "warm" as <i>not relevant</i>, because a fiery temper isn\'t either warm or the opposite of warm.</p>'+
  		'<p>Rate "cold" as <i>not relevant</i>, because a fiery temper isn\'t either cold or the opposite of cold</p>',

		
  	
  	labels: ["not relevant","relevant"],
  	prompt: ["intense","calm","warm","cold"],
  	response_ends_trial: true,
    // choices: shuffle([conds[condition]["literal"][0],conds[condition]["metaphorical"][0]]),
    // prompt: "<br>Which adjective is better?<br>"
	}

// var exp0 = {
//     type: 'survey-multi-choice',
//     questions: [
//     	{prompt: 'Sentence:'+topics[0], options: [items[0],items[0]], required:true,},
//     	{prompt: 'Sentence:'+topics[0], options: [items[0],items[0]], required:true,}, 
//     	]
// 	}
// var i = 0
// var exp0 = {
//     type: 'html-multi-slider-response',
//     stimulus:
//   	'Sentence:'+topics[i],
//   	labels: shuffle([items[i][0],items[i][1]]),
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

var survey = {
  type: 'survey-text',
  questions: [{prompt: "Use this box to give any feedback you have on the experiment, and anything that confused you."},],
};

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
timeline.push(survey)
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
