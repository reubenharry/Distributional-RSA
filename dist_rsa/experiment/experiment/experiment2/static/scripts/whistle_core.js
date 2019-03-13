
// IMPORTANT STATIC VARIABLES
var frequencyRange = [300, 1200];
var filterDepth = 4;
var neutralPos = 0.0;

// var data_array, record;
var $flute = undefined,
	$container = undefined,
	topM, height,
	currentFrequency, relPos, innerPos,
	triggerTime = 0,
	lastTime = 0,
	keyisdown = false,
	demoMode = false, //enable demo mode
	rerecording = false; //enable or disable ability to rerecord

var timer = 0;
var data_array = [],
	all_data = [];
var record = false,
	attempts = 0;

var endKey = 81, //'q' key
	repeatKey = 82, //'r' key
	onKey = 32;   //spacebar

var instrument = [];
var synth, noise, filter;


var saveDataPoint = function(e) {
	if (record) {
		data_array.push({
			e : e.type,
			a : keyisdown,
			t : e.timeStamp,
			f : currentFrequency,
			y : relPos,
			p : innerPos
		});
		// TODO
		// maybe add another datapoint (mouseover)
		// then event == keyup (for plotting)
	}
}


// define scale for feedback
var performanceBounds = [50, 300]
// good: range 0-50    0.9
// acceptable: range 50-250 0.5
// mh.. range 250-500 0.2
// virtually 0: 500+

var play_disabled = false,
	inactime_time = 5000,
	check_interval = 10,
	inactivate = true;

function checkForInactive() {
	timer = timer + check_interval;
    if (timer >= inactime_time && !play_disabled && !keyisdown && inactivate) {
    	play_disabled = true;
    	console.log('Play disabled!');
    	show_inactive_box(300);
    } else if (timer < inactime_time && play_disabled) {
    	play_disabled = false;
    	console.log('Play enabled!');
    	hide_inactive_box();
    } else {
    }
    setTimeout("checkForInactive();", check_interval);
}

// $(document.body).on("load", checkForInactive());

function show_inactive_box(fadeTime) {
	$('#inactive_wrapper').css('visibility','visible').hide().fadeIn(fadeTime);
	$('#inactive_box').css('visibility','visible').hide().fadeIn(fadeTime);
	$('#flute-container').addClass('blur');
}
function hide_inactive_box() {
	$('#inactive_wrapper').css('visibility','hidden');
	$('#inactive_box').css('visibility','hidden');
	$('#flute-container').removeClass('blur');
}







var frequencyFromPosition = function(relPos) {
	lowerPosBound = frequencyRange[0]/frequencyRange[1]
	multiple = (1-relPos)*(1-lowerPosBound)+lowerPosBound;
	f = frequencyRange[0] * 1/multiple;
	return f
}

var changeFrequency = function(e) {
	if (e.timeStamp > lastTime || e.type != 'mousemove') { //only track every timeStamp once
		timer = 0;
		lastTime = e.timeStamp;
		mouseY = e.pageY;
		innerPos = Math.min(Math.max(mouseY, topM), topM+height) - topM;
		relPos = innerPos / height;
		//move plunger
		$( "#piston, #plunger" ).css('background-position-y', innerPos + 'px');
		// change frequency
		currentFrequency = frequencyFromPosition(relPos)
		synth.frequency.linearRampToValueAtTime(currentFrequency, "+0.00001");
		filter.frequency.linearRampToValueAtTime(currentFrequency*filterDepth, "+0.00001");
		saveDataPoint(e);
	}
};

// prevent use of space bar
window.addEventListener('keydown', function(e) {
  if(e.keyCode == 32 && e.target == document.body) {
	e.preventDefault();
  }
});

// special functions for ending the trial (in experiment mode)
var endTrial = function(e) {
	// entTrial is not active in demo mode!
	if (e.keyCode == repeatKey && !keyisdown && !demoMode && rerecording) {
		alert('Your recording has been deleted, please start over.');
		timer = 0;
		data_array = trimEnds(data_array);
		attempts += 1;
		all_data.push(data_array);
	}

	if (e.keyCode == endKey && !keyisdown && !demoMode) {
		timer = 0;
		data_array = trimEnds(data_array);
		if (data_array.length==0 && !$('#promptText', this).text()=='') {
			alert($('#promptText', this).text());
			return
		} else {
			stopRecording();
			disablePlay();
			var callback = e.data['onEnd'];
			callback();
		}
	}
}

var noteOn = function(e) {
	if (e.keyCode == onKey && $('#flute-container:hover').length != 0 && !keyisdown) {
		synth.triggerAttack(currentFrequency);
		noise.triggerAttackRelease(0.1);
		triggerTime = performance.now(); //maybe something with higher resolution is needed?
		keyisdown = true;
		saveDataPoint(e);
	}
}

var noteOff = function(e) {
	if(e.keyCode == onKey && keyisdown) {
		// wait a minimum of delta T
		timer = 0;
		delta = Math.max(0, synth.envelope.attack - (
			performance.now() - triggerTime)/1000.0)*1000.0;
		setTimeout(
		  function() {
			synth.triggerRelease();
			keyisdown = false;
			saveDataPoint(e);
		  }, delta);
	}
}

var updateDOMVariables = function() {
	// update DOM variables (get current values)
	$(document).ready(function() {
		$flute = document.querySelector("#piston");
		$container = document.querySelector("#flute-container");
		topM = $( "#piston" ).position().top + 10;
		height = $( "#piston" ).height() - 30;
	});
}

var enablePlay = function(callback=function(){}) {
	timer = 0;
	//generate and bind instrument instance
	setInstrumentVariables(generateInstrument());
	updateDOMVariables();
	_bindInstrumentHandlers(callback);
}

var disablePlay = function() {
	_unbindInstrumentHandlers();
}

var _bindInstrumentHandlers = function(callback=function(){}) {
	$(document).ready( function() {
		$( "#flute-container" ).on('mousemove', changeFrequency);
		$(document).on('keydown', noteOn);
		$(document).on('keyup', noteOff);
		$(document).on('keydown', {onEnd : callback}, endTrial);
	});
}

var _unbindInstrumentHandlers = function() {
	$( "#flute-container" ).off('mousemove', changeFrequency);
	$(document).off('keydown', noteOn);
	$(document).off('keyup', noteOff);
	$(document).off('keydown', endTrial);
}

var startRecording = function() {
	record = true;
	data_array = [];
	all_data = [];
	attempts = 0;
}

var stopRecording = function() {
	record = false;
	// utterance is now temporarily saved in data_array
	data_array = trimEnds(data_array);
	for (var i = 0; i < all_data.length; i++) {
		all_data[i] = trimEnds(all_data[i]);
	}
}

var trimEnds = function(data_array){
	if (data_array.length == 0) {
		return data_array;
	}

	var firstFound = false,
		startIndex = 0,
		endIndex = data_array.length-1;
	for (var i = 0; i < data_array.length; i++) {
		if (data_array[i].e == 'keydown' && !firstFound) {
			firstFound = true;
			startIndex = i;
		} else if (data_array[i].e == 'keyup') {
			endIndex = i+1;
		}
	}
	if (firstFound == false) {
		// signal track is empty (no data)
		return []
	} else {
		data_array = data_array.slice(startIndex, endIndex)
		t0 = data_array[0].t;
		for (var i = 0; i <= data_array.length-1; i++) {
			data_array[i].t -= t0;
		}
		return data_array
	}
}

var addPadding = function(data_array, return_to_neutral=true) {

	var sigmoidRampFromTo = function(p0, p1, t0, t1, f, res=2) {
		function linspace(a, b, n) {
			ret = Array(n), n--;
			for (i = n; i >= 0; i--) {
				ret[i] = (i * b + (n - i) * a) / n
			}
			return ret
		}
		function logistic(x, k=10) {
			return 1/(1+Math.exp(-k*x))
		}
		n = Number((t1 - t0) / res)
		xx = linspace(-1, 1, n)
		pos_array = Array(n)
		for (var i = 0; i < xx.length; i++) {
			pos_array[i] = logistic(xx[i])*(p1-p0) + p0
		}
		time_array = linspace(t0, t1, n)
		padding_array = new Array();
		for (var i = 0; i < Number((t1 - t0)/res); i++) {
			padding_array.push({
				e: 'mousemove', a: false, f: f, p: 0,
				y: pos_array[i], t: time_array[i]
			});
		}
		return padding_array
	}

	var silenceDur = 500,
		paddingDurMs = 400,
		paddingRes = 5;

	var startPos = data_array[0].y,
		startTime = data_array[0].t,
		startF = data_array[0].f,
		endPos = data_array.slice(-1)[0].y,
		endTime = data_array.slice(-1)[0].t,
		endF = data_array.slice(-1)[0].f;

	var before = sigmoidRampFromTo(
		neutralPos, startPos, startTime-paddingDurMs-silenceDur, startTime-silenceDur, startF, paddingRes);
	var after = sigmoidRampFromTo(
		endPos, neutralPos, endTime, endTime+paddingDurMs, endF, paddingRes);
	var silence = sigmoidRampFromTo(
		startPos, startPos, startTime-silenceDur, startTime, startF, paddingRes);
	var silenceEnd = sigmoidRampFromTo(
		endPos, endPos, endTime, endTime+silenceDur*2, endF, paddingRes);

	if (return_to_neutral) {
		data_array = before.concat(silence, data_array, after);
	} else {
		data_array = before.concat(silence, data_array, silenceEnd);
	}
	t0 = data_array[0].t;
	for (var i = 0; i <= data_array.length-1; i++) {
		data_array[i].t -= t0
	}
	return data_array
}

var playBackRecording = function(data_array, feedback=true, return_to_neutral=true, callback=function(){}) {

	//in case no instrument is active yet
	setInstrumentVariables(generateInstrument());
	updateDOMVariables();
	_unbindInstrumentHandlers();

	data_array = addPadding(trimEnds(data_array), return_to_neutral);

	$(document).ready(function() {
		schedulePlayback(data_array, feedback);
		//total time before executing callback
		t_total = data_array[data_array.length-1].t - data_array[0].t;
		window.setTimeout(function(){
			callback();
		}, t_total);
	});
}

// function to play a recorded utterance (also for offline rendering)
var schedulePlayback = function(data_array, visual_feedback) {

	// helper function to change position of plunger on screen
	var changePlungerPosition = function(t, p) {
		if (!visual_feedback) { return }
		window.setTimeout(function(){
			$("#piston, #plunger").css("background-position-y", +p+"px");
		}, t*1000.0);
	};

	var t0 = data_array[0].t;
	startTime = performance.now();
	for (var i = 0; i < data_array.length; i++) {
		nowTime = performance.now() - startTime;
		t_num = ( (data_array[i].t - t0 - nowTime)/1000.0 );
		t = ("+" + t_num.toString()).substring(0, 8);

		synth.frequency.linearRampToValueAtTime(data_array[i].f, t);
		filter.frequency.linearRampToValueAtTime(data_array[i].f*filterDepth, t);
		changePlungerPosition(t, height*data_array[i].y);
		if (data_array[i].e == 'keydown') {
			synth.triggerAttack(data_array[i+1].f, t);
			noise.triggerAttackRelease(0.1, t);
		} else if (data_array[i].e == 'keyup') {
			t_rel = ("+" + (t_num+0.01).toString()).substring(0, 6);
			synth.triggerRelease(t_rel);
		}
	}
}

var downloadWav = function(data_array, filename) {

	data_array = addPadding(trimEnds(data_array));
	t_total = (data_array[data_array.length-1].t - data_array[0].t)/1000.0;

	return Tone.Offline(function() {
		//only nodes created in this callback will be recorded
		var save_instrument = generateInstrument();
		setInstrumentVariables(save_instrument);
		schedulePlayback(data_array, false);

	}, t_total).then(function(buffer) {
		setInstrumentVariables(generateInstrument());
		var audioBuffer = buffer.get();
		audioBufferToWaveBlob(buffer).then(function(blob){
			saveAs(blob, filename);
		})
	});
}

var downloadJson = function(data_array, fileName) {
	var utterance = JSON.stringify(data_array),
		blob = new Blob([utterance], {type: "text/plain;charset=utf-8"});
	saveAs(blob, fileName);
}


var setInstrumentVariables = function(instrument) {
	synth = instrument[0];
	noise = instrument[1];
	filter = instrument[2];
}

// INSTRUMENT SPECIFICATION
var generateInstrument = function() {
	var vibrato = new Tone.Vibrato({
		frequency: 1,
		depth : 0.5
	}).toMaster();

	var filter = new Tone.Filter({
		type : "lowpass",
		frequency : 440,
		Q: 4
	}).connect(vibrato);

	var synth = new Tone.Synth({
		oscillator : {
			type : 'triangle'
		},
		envelope : {
			attack : 0.05,
			decay : 0.2,
			sustain : 0.65,
			release : 0.05
		},
		freuency : 440,
		volume : -10
	}).connect(filter);

	var noise = new Tone.NoiseSynth({
		noise : {
			type : 'pink'
		},
		envelope : {
			attack : 0.005,
			sustain : 1.0,
			release : 1.0
		},
		volume : -60
	}).toMaster();
	return [synth, noise, filter]
}



// Similarity computations
function scoreDistance(distance) {
	if (distance < performanceBounds[0]) {
		return 0
	} else if (distance < performanceBounds[1]) {
		return 1
	} else {
		return 2
	}
}

function scoreProbability(distance, a=4, b=1.5, c=0.01) {
	return a/(c*distance**b+a)
}

function compareDistance(firstUtterance, secondUtterance) {
	  var interpolate_array = function(sourceArray) {
		if (sourceArray.length==0) {
		  return [0]
		}
		newArray = []
		runningCount = 0;
		for (var i = 0; i < sourceArray[sourceArray.length-1]['t']; i++) {
		  if (Math.round(sourceArray[runningCount]['t']) == i) {
			newArray.push(sourceArray[runningCount]['y']);
			runningCount++;
		  } else {
			newArray.push(newArray[newArray.length-1]);
		  }
		}
		return newArray
	  }

	  // make sure the arrays start at t=0
	  var dtw = new DTW();
	  var cost = dtw.compute(
		interpolate_array(trimEnds(firstUtterance)),
		interpolate_array(trimEnds(secondUtterance)));
	  // console.log('Cost: ' + cost);
	  return cost
}

