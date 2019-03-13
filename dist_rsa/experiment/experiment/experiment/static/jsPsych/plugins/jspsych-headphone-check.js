/** (July 2012, Erik Weitnauer)
The html-plugin will load and display an external html pages. To proceed to the next, the
user might either press a button on the page or a specific key. Afterwards, the page get hidden and
the plugin will wait of a specified time before it proceeds.

documentation: docs.jspsych.org
*/

jsPsych.plugins['headphone-check'] = (function() {

	var plugin = {};

	plugin.info = {
		name: 'headphone-check',
		description: '',
		parameters: {
			url: {
				type: jsPsych.plugins.parameterType.STRING,
				pretty_name: 'URL',
				default: undefined,
				description: 'The url of the external html page'
			},
			cont_btn: {
				type: jsPsych.plugins.parameterType.STRING,
				pretty_name: 'Continue button',
				default: null,
				description: 'The button to continue to the next page.'
			},
			get_psiturk_variable: {
				type: jsPsych.plugins.parameterType.FUNCTION,
				pretty_name: 'Psiturk variable',
				default: function() { return true; },
				description: 'Function to retrieve the psiturk variable'
			}
		}
	}

	plugin.trial = function(display_element, trial) {

		var url = trial.url;

		load(display_element, url, function() {
			var t0 = (new Date()).getTime();

			console.log('loading complete');


			// N = trial.N
			N = 3
			var HC_key = [];
			var HC_version = [];
			var HC_but = [];

			// randomize the stimuli
			for(i = 0; i < N; i++){
					jj = Math.floor(Math.random() * 3) + 1;
					HC_key[i] = jj.toString();
					jj = Math.floor(Math.random() * 2) + 1 ;
					HC_version[i] = jj.toString();
					HC_but.push("antiphase_HC_".concat(HC_key[i],"_",HC_version[i]));
			}

			// console.log(HC_key);
			// console.log(HC_version);
			// console.log(HC_but);

			// N = number of test stimuli
			for (var i = 0; i < N; i++) {
				html = '';
				html += '<div class="row question">';
				html += '<div class="col-md-1"></div>';
				html += '<div class="col-md-3" id="hpc_playbutton_'+ i.toString() +'">';
				html += '<button id="antiphase_HC_b_'+ i.toString() +'" type="button" value="Calibration" class="btn btn-primary" data-choice="'+i+'">Click to play sounds</button>';
				html += '</div>';
				html += '<div class="col-md-4">Which sound was softest/quietest?</div>';
				html += '<div class="col-md-2"><select id="hpc_select_'+ i.toString() +'" name="hpc_n'+i.toString()+'">';
				html += '<option value="1">First</option>';
				html += '<option value="2">Second</option>';
				html += '<option value="3">Third</option>';
				html += '<option hidden disabled selected value> -- </option>';
				html += '</select></div>';
				html += '</div>';

				$("#headphone_check").append(html);
				// var target = display_element.querySelector('#headphone_check');
				// var div = document.createElement('div');
				// div.innerHTML = html.trim();
				// while (div.firstChild) {
				// 	target.appendChild(div.firstChild);
				// }

			}
			for (var i = 0; i < N; i++) {
				display_element.querySelector('#antiphase_HC_b_' + i).addEventListener('click', function(e){
					var choice = e.currentTarget.getAttribute('data-choice'); // don't use dataset for jsdom compatibility
					console.log(HC_but[choice]);
					var clip = document.getElementById(HC_but[choice]);
					clip.play();
				});
			}

			var finish = function() {
				correct_responses = 0;
				responses = [];
				incomplete = false;
				for (var i = 0; i < N; i++) {
					var val = $('#hpc_select_'+ i.toString() +' option:selected').val();
					if (val == HC_key[i]){ correct_responses++ };
					if (val == ''){
						alert("Please respond to all items!");
						incomplete = true;
						break
					};
					responses.push(val);
				}

				if (incomplete == false) {
					var trial_data = {
						rt: (new Date()).getTime() - t0,
						url: trial.url
					};
					display_element.innerHTML = '';
					trial_data.headphonecheck_passed = (correct_responses == N)
					jsPsych.finishTrial(trial_data);
				};
			}

			if (trial.cont_btn) { display_element.querySelector('#'+trial.cont_btn).addEventListener('click', finish); }
		}); //end load
	}; //end plugin

	// helper to load via XMLHttpRequest
	function load(element, file, callback){
		var xmlhttp = new XMLHttpRequest();
		xmlhttp.open("GET", file, true);
		xmlhttp.onload = function(){
				if(xmlhttp.status == 200 || xmlhttp.status == 0){ //Check if loaded
						element.innerHTML = xmlhttp.responseText;
						callback();
				}
		}
		xmlhttp.send();
	}

	return plugin;
})();
