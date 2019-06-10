/**
 * Experiment to collect productions from people on Mturk
 * - Matthias Hofer
 *
 * code v6 (different experiment)
 *
 * uses custom plugins
 *
 * documentation: mit.github/mhofer
 *
 **/

// TODO: get these variables from a global settings file!

/* experiment-wide variables */
var USING_PSITURK,
	DEBUG = false,
	COUNTER = 1,
	N_TRIALS = 30,
	BASE_PAY = 0.30; //not actually used here but for reference


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


/* create jspsych timeline */
var timeline = [];



var exp_p_task_description = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: "<p class='bounded left-text'> In this survey, you'll see two text message conversations. Read the conversations carefully, and then answer the question below. Press (q) to continue.</p>"
};

var exp_c_task_description = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: "<p class='bounded left-text'> In this survey, you'll see several panels of a comic strip. Read the panels carefully, and then answer the question below. Press (q) to continue.</p>"
};










var intro_c1 = "expcintro1"
var intro_c2 = "expcintro2"
var intro_c3 = "expcintro3"
var intro_c4 = "expcintro4"
var intro_c5 = "expcintro5"
var intro_c6 = "expcintro6"
// var intro_c6 = (Number(condition)==0||Number(condition)==2)?"expcintro6_cond1":"expcintro6_cond2"
// var test_c = (Number(condition)==0||Number(condition)==2)?"expctest1":"expctest2"
var test_c = "expctest1"

var exp_c = {
    type: 'html-slider-response',



    stimulus: "<p class='bounded left-text'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c3+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c4+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c5+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c6+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+test_c+".png' alt='image'>"
    +"<span style='display: block; font-size: 12px;'></span><br \> </p>",
    labels: ["<b><i>Somewhere Else</i></b>","<b><i>Olives</i></b>",],
    prompt: "<p>When Margaret says she's going to a cafe in the last panel, how likely do you think it is that she means <i>Olives</i>? <b>Mark your answer on the slider above.</b></p>",
	}

// var condition=0

var intro_p = ((Number(condition)==0)||(Number(condition)==1))?"exppintro1":"exppintro2"
var test_p = ((Number(condition)==0)||(Number(condition)==3))?"expptest1":"expptest2"
var name_p = ((Number(condition)==0)||(Number(condition)==1))?"Arthur":"Sue"


var exp_p_intro = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: '<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b>  Later that day:  </b>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
	+'</div>'+"<p>Read the conversations carefully and press (q) to continue to a question.</p>",
};

var exp_p = {
    type: 'html-slider-response',

    stimulus:

  	'<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b>  Later that day:  </b>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
	+'</div>',

    // stimulus: "<p class='bounded left-text'> <b><big> Time: 12:30PM </big></b>" 
    // 	+"<img style='height: 10%;' src='/static/resources/exp6_rating/stimuli/testcondexp2phone3.png' alt='image'>"
    // 	+"<img style='height: 10%;' src='/static/resources/exp6_rating/stimuli/testcondexp2phone3.png' alt='image'>"
    // 	+"<span style='display: block; font-size: 12px;'></span><br \> </p>",
    labels: ["<b><i>Somewhere Else</i></b>","<b><i>Olives</i></b>",],
    prompt: "<p> When Margaret says she's going to a cafe in the conversation on the right, how likely do you think it is that she means <i>Olives</i>? <b>Mark your answer on the slider above.</b></p>",
	
	}

// var exp2p_cond4 = {
//     type: 'image-button-response',
//     stimulus: '/static/resources/exp6_rating/stimuli/testcondexp2phone4.png',
//     choices: ["Sally","Someone else"],
//     prompt: "<p><b>Condition 4: Phone: </b>Which friend is Margaret talking about?</p>"
// 	}

// var exp2c_button_trial = 

// 	Number(condition)==1?
// 	exp2c_cond1
// 	:
// 	Number(condition)==2?
// 	exp2c_cond2
// 	:
// 	exp2c_cond3


var page_1_options = ["Olives","Starbucks","Somewhere else"]
var exp_p_sanity_check = 

	
	{
	type: 'html-button-response',
    stimulus: '<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b>  Later that day:  </b>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
	+'</div>',
    choices: page_1_options,
    prompt: "<p>What is the name of the cafe that "+name_p+" got a sandwich at?</p>"
 	}

var exp_c_sanity_check = 

	
	{
	type: 'html-button-response',
    stimulus: '',
    choices: page_1_options,
    prompt: "<p>Where were Arthur and Sue originally planning to get lunch?</p>"
	}

 //    type: 'html-button-response',
 //    stimulus: 
 //    questions: [{
 //    	// prompt: "Which of these people <b>wasn't</b> included in the messages on the last page?", options: page_1_options, required:true,}]
 //    	prompt: "Where were Arthur and Sue originally planning to get lunch?", options: page_1_options, required:true,}]
	// }


var survey = {
  type: 'survey-text',
  questions: [{prompt: "Briefly explain why you gave the rating on the slider that you did."},],
};







var end_exp = {
	type: "html-keyboard-response",
	stimulus: "<p id='bounded'>Thank you for participating!<br\><br\>Press any key to end the survey.</p>",
	on_load: function() {
		spacebar_active = false;
	}
};

/******************************
 *
 *	Timeline composition
 *
 *
 *
 *
 *
 ******************************/

// timeline.push(exp_c_task_description);
// timeline.push(exp_c);
// timeline.push(exp_c_sanity_check);

timeline.push(exp_p_task_description);
timeline.push(exp_p_intro)
timeline.push(exp_p);
timeline.push(exp_p_sanity_check);

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
