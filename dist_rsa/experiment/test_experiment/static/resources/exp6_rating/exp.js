
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


/* create jspsych timeline */
var timeline = [];

// var condition=2


var exp_p_task_description = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: "<p class='bounded left-text'> <b>Please make sure this window is fullscreen, if possible.</b> In this survey, you'll see two text message conversations. Read the conversations carefully, and then answer the questions below. Press (q) to continue.</p>"
};

var divider = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: "<p class='bounded left-text'> Press (q) to continue.</p>"
};




var intro_p = ((Number(condition)==0)||(Number(condition)==1))?"new_exppintro1":"new_exppintro2"
// var test_p = ((Number(condition)==0)||(Number(condition)==3))?"new_expptest1":"new_expptest2"

var test_p = ((Number(condition)==0)||(Number(condition)==2))?"multi_expptest1":"multi_expptest2"

var name_p = ((Number(condition)==0)||(Number(condition)==1))?"Arthur":"Sue"


var name_c = ((Number(condition)==0)||(Number(condition)==2))?"Arthur":"Sue"

// var between = (Number(condition)==0)?"betweencond0":
	// (Number(condition)==1)?"betweencond1":
	// (Number(condition)==2)?"betweencond2":
	// "betweencond3"

// var intro_variable = ((Number(condition)==0)||(Number(condition)==3))?"Arthur again":"Sue"

var exp_p_intro = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: '<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b> The next day: </b>'
    // + '<a class="icons"><img height=50 src="/static/resources/exp6_rating/stimuli/'+between+'.png"></a>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
	+'</div>'+"<p>Read the conversations carefully and press (q) to continue to a question.</p>",
};


// var exp_p = {
//     type: 'html-slider-response',

//     stimulus:

//   	'<div class="nav3" style="height:505px;">'
//     +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
//     +'<b> The next day: </b>'
//     // + '<a class="icons"><img height=50 src="/static/resources/exp6_rating/stimuli/'+between+'.png"></a>'
//     +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
// 	+'</div>'
// 	+"<p><br>How likely is it that Margaret got the cookie at Flour?</p>",

//     // stimulus: "<p class='bounded left-text'> <b><big> Time: 12:30PM </big></b>" 
//     // 	+"<img style='height: 10%;' src='/static/resources/exp6_rating/stimuli/testcondexp2phone3.png' alt='image'>"
//     // 	+"<img style='height: 10%;' src='/static/resources/exp6_rating/stimuli/testcondexp2phone3.png' alt='image'>"
//     // 	+"<span style='display: block; font-size: 12px;'></span><br \> </p>",
//     labels: ["<b>Unlikely</b>","<b>Likely</b>"],
//     prompt: ""

	
// 	}

var exp_p = {
    type: 'html-button-response',

    stimulus:

  	'<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b> The next day: </b>'
    // + '<a class="icons"><img height=50 src="/static/resources/exp6_rating/stimuli/'+between+'.png"></a>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
	+'</div>'+"<br><p>Do you think Margaret got the cookies from Flour?</p>",
    choices: ["Yes","No"],
    prompt: "<br><br>"
	
	}



// <b>Mark your answer on the slider above.</b></p>",

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


var page_1_options = ["Yes","No","Unclear"]

var exp_p_sanity_check_1 = 

	
	{
	type: 'html-button-response',
    stimulus: '<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b> The next day: </b>'
    // + '<a class="icons"><img height=50 src="/static/resources/exp6_rating/stimuli/'+between+'.png"></a>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
    +'</div>'+"<br>Did Margaret go with "+name_p+" when she got the cookies?",
    choices: ["Yes","No"],
    prompt: "<br><br>"
	}

var exp_p_sanity_check_2 = 

	
	{
	type: 'html-button-response',
    stimulus: '<div class="nav3" style="height:505px;">'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+intro_p+'.png"></a>'
    +'<b> The next day: </b>'
    // + '<a class="icons"><img height=50 src="/static/resources/exp6_rating/stimuli/'+between+'.png"></a>'
    +'<a class="icons"><img height=500 src="/static/resources/exp6_rating/stimuli/'+test_p+'.png"></a>'
    +'</div>'+"<br>Why did Margaret get cookies?",
    choices: ["Somebody's birthday","Request","We don't know"],
    prompt: "<br><br>"
	}



var survey_p = {
  type: 'survey-text',
  questions: [{prompt: "Briefly explain why you gave the answer that you did to the previous question."},],
};





















var exp_c_task_description = {
	type: "html-keyboard-response",
	choices: ['q'],
	stimulus: "<p class='bounded left-text'> <b>Please make sure this window is fullscreen, if possible.</b> In this survey, you'll see several panels of a comic strip. Read the panels carefully, and then answer the questions below. Press (q) to continue.</p>"
};

var intro_c1 = "new_expcintro1"
var intro_c2 = "new_expcintro2"
// var test_c = (Number(condition)==0||Number(condition)==2)?"new_expctest1":"new_expctest2"

var test_c = (Number(condition)==0||Number(condition)==1)?"new_expctest1":
	(Number(condition)==2||Number(condition)==3)?"new_expctest2":
	// (Number(condition)==2||Number(condition)==3)?"multi_expctest1":
	false
	// "new_expctest2"

// var test_c = (Number(condition)==0||Number(condition)==2)?"multi_expctest1":"multi_expctest2"

// var exp_c_test = {
//     type: 'html-slider-response',



//     stimulus: "<p class='bounded left-text'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+test_c+".png' alt='image'>"
//     +"<span style='display: block; font-size: 12px;'></span><br \> </p>",
//     labels: ["<b><i>Definitely a different cafe</i></b>","<b><i>Definitely Flour</i></b>",],
//     prompt: "<p> <b>Rate on the slider above</b> how much you think that Margaret got the cookies from <i>Flour</i> or from somewhere else.</p>",
// 	}

var exp_c_test = {
    type: 'html-button-response',



    stimulus: "<p class='bounded left-text'>"
    // +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+test_c+".png' alt='image'>"
    +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
    +"<br><p>Do you think Margaret got the cookies from Flour?</p>",
    choices: ["Yes","No"],
    prompt: "<br><br>",
	}

// var exp_c_test1 = {
//     type: 'html-button-response',



//     stimulus: "<p class='bounded left-text'>"
//     // +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+"new_expctest1"+".png' alt='image'>"
//     +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
//     +"<br><p>Do you think Margaret got the cookies from Flour?</p>",
//     choices: ["Yes","No"],
//     prompt: "<br><br>",
// 	}

// var exp_c_test2 = {
//     type: 'html-button-response',



//     stimulus: "<p class='bounded left-text'>"
//     // +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+"new_expctest2"+".png' alt='image'>"
//     +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
//     +"<br><p>Do you think Margaret got the cookies from Flour?</p>",
//     choices: ["Yes","No"],
//     prompt: "<br><br>",
// 	}

// var exp_c_test3 = {
//     type: 'html-button-response',



//     stimulus: "<p class='bounded left-text'>"
//     // +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+"multi_expctest1"+".png' alt='image'>"
//     +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
//     +"<br><p>Do you think Margaret got the cookies from Flour?</p>",
//     choices: ["Yes","No"],
//     prompt: "<br><br>",
// 	}

// var exp_c_sanity_check = 

	
// 	{
// 	type: 'html-button-response',
//     stimulus: "<p class='bounded left-text'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
//     +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+test_c+".png' alt='image'>"
//     +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
//     +"Which two people work in the same office?",
//     choices: ["Margaret and Arthur","Arthur and Sue","Sue and Margaret"],
//     prompt: "<br><br>"
// 	}

var exp_c_sanity_check_1 = 

	
	{
	type: 'html-button-response',
    stimulus: "<p class='bounded left-text'>"
    // +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+test_c+".png' alt='image'>"
    +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
    +"Did Margaret go with "+name_c+" when she got the cookies?",
    choices: ["Yes","No"],
    prompt: "<br><br>"
	}

var exp_c_sanity_check_2 = 

	
	{
	type: 'html-button-response',
    stimulus: "<p class='bounded left-text'>"
    // +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c1+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+intro_c2+".png' alt='image'>"
    +"<img style='height: 60%;' src='/static/resources/exp6_rating/stimuli/"+test_c+".png' alt='image'>"
    +"<span style='display: block; font-size: 12px;'></span><br \> </p>"
    +"Why did Margaret get "+name_c+" cookies?",
    choices: ["Birthday present","Request","Unknown"],
    prompt: "<br><br>"
	}

 //    type: 'html-button-response',
 //    stimulus: 
 //    questions: [{
 //    	// prompt: "Which of these people <b>wasn't</b> included in the messages on the last page?", options: page_1_options, required:true,}]
 //    	prompt: "Where were Arthur and Sue originally planning to get lunch?", options: page_1_options, required:true,}]
	// }




var survey_c = {
  type: 'survey-text',
  questions: [{prompt: "Briefly explain why you gave the answer that you did to the previous question."},],
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


// timeline.push(exp_p_task_description);
// timeline.push(exp_p_intro)
// timeline.push(exp_p_sanity_check_1);
// timeline.push(divider);
// timeline.push(exp_p_sanity_check_2);
// timeline.push(divider);
// timeline.push(exp_p);
// timeline.push(survey_p);


timeline.push(exp_c_task_description);
// timeline.push(exp_c_sanity_check_1);
// timeline.push(divider);
timeline.push(exp_c_sanity_check_2);
timeline.push(divider);
timeline.push(exp_c_test);
// timeline.push(exp_c_test1);
// timeline.push(exp_c_test2);
// timeline.push(exp_c_test3);
timeline.push(survey_c)







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
