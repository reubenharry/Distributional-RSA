/**
 * jspsych-flute-reproduction
 * Matthias Hofer
 *
 * plugin for displaying the flute and playing
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["flute-reproduction"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "flute-reproduction",
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: '',
        description: 'The utterance to be played'
      },
      useGlobalStimulusID: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Source',
        default: false,
        description: 'If true use global variable to determine current stimulus'
      },
      text: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: undefined,
        description: 'The text to be displayed'
      },
      promptText: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: 'You must record a whistle!',
        description: 'The text to be displayed'
      },
      feedbackMode: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Feedback',
        default: 'none',
        description: 'Whether or not to give DTW-based feedback.'
      },
    }
  }


  plugin.trial = function(display_element, trial) {

    // variables for feedback
    var feedback_texts_learning = [
      "Well done!",
      "OK, but there's room for improvement!",
      "Not similar enough!"
    ];
    var feedback_texts_test = [
      "Success! The computer guessed it right.",
      "____",
      "The computer was not able to identify the right color."
    ];
    var color_categories = [
      'green', 'orange', 'red'
    ];
    var feedback_category,
        feedbackDisplayDuration;

    // display whistle
    var html =  ''
    html += '<div id="feedback-wrapper">' + trial.text + '</div>';
    html += '<div id="flute-container">'

    html += '<div id="inactive_wrapper">'
    html += '<div id="inactive_box">'
    html += '<p>The instrument is inactive.</p>'
    html += '<p>Mouse over to activate.</p>'
    html += '</div>'
    html += '</div>'

    html += '<div id="left_container">';
    html += '<div id="piston"></div><div id="body"></div></div>';
    html += '<div id="right_container"><div id="plunger"></div></div></div>';
    // html += '<p></p>'
    html += '<div id="promptText" style="display: none;">'+ trial.promptText+ '</div>';
    display_element.innerHTML = html;

    var start_time = Date.now();
    // store response
    var response = {
      rt: null,
      stimulus: [],
      earlier_attempts: [],
      attempts: 0,
      data: []
    };

    // Determine source of the stimulus variable
    if (trial.useGlobalStimulusID) {
      console.log('Use global ID');
      trial.stimulus = stimuli[CURRENT_STIMULUS_ID].stimulus;
    }

    if (trial.stimulus == '') {
      // no reference stimulus was passed
      console.log('No reference stimulus passed!');
      response.stimulus = [];
      enablePlay(end_trial);
      startRecording();
    } else {
      //load json file and play
      console.log('Reference stimulus available.');
      loadJSON(trial.stimulus, function(jsonFile) {
        var reference_utterance = JSON.parse(jsonFile);
        response.stimulus = reference_utterance;
        enablePlay(end_trial);
        startRecording();
      });
    }


    function end_trial() {
      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      var end_time = Date.now();
      var rt = end_time - start_time;

      response.data = data_array;
      response.rt = rt;
      response.attempts = attempts;
      response.earlier_attempts = all_data;

      // gather the data to store for the trial
      var trial_data = {
        "rt": response.rt,
        "production" : response.data,
        "earlier_productions" : response.earlier_attempts,
        "attempts" : response.attempts,
        "feedback" : trial.feedbackMode
      };


      if (trial.feedbackMode != 'none') {
        console.log('Feedback given.');
        // mode != none => feedback will be given

        // $('#flute-container *').css('background-image', 'none');

        if (FULL_COMMUNICATION_FEEDBACK) {
          feedbackDisplayDuration = 3000;
        } else {
          feedbackDisplayDuration = 2000;
        }

        var distances = [],
            distance = 0,
            itemsProcessed = 0;

        var changeFeedback = function(text, callback) {
          $('#feedback-wrapper p').text(text);
          $('#feedback-wrapper').css('background-color', 'black');
          $('#feedback-wrapper').css('color', 'white');
          callback();
        }

        if (trial.feedbackMode == 'all') {


          function show_feedback () {
            console.log("Distances", distances);

            // get inferred referent and stimulus
            inferred_stimulus_SRC = stimuli[argMin(distances)].stimulus;
            inferred_referent_SRC = stimuli[argMin(distances)].referent;

            // get ground truth referent;
            for (var i = 0; i < stimuli.length; i++) {
              if (trial.stimulus==stimuli[i].stimulus) { break; }
            }; true_referent_SRC = stimuli[i].referent;

            console.log("Inferred", inferred_referent_SRC);
            console.log("True", true_referent_SRC);

            var computer_corret = inferred_referent_SRC==true_referent_SRC;
            trial_data["correct"] = computer_corret;

            feedback_category = computer_corret ? 0 : 1;
            if (feedback_category==0) { SCORE++; } //TODO: implement this via data.variable

            all_feedback_text = ["Success!", "Failure."]
            alt_feedback_text = ["", " instead"]
            html =  ''
            html += '<div>'
            html += '<div id="feedback-header">'+all_feedback_text[feedback_category]+'</div>'

            if (FULL_COMMUNICATION_FEEDBACK) {
              // html += '<p>The computer guessed the right item.</p>'
              alt_feedback_text = ["", " instead"]
              html += '<div id="feedback-wrapper">The computer guessed the following item'+alt_feedback_text[feedback_category]+'.</div>';
              html += '<img src="'+inferred_referent_SRC+'" class="feedback_referent">'
            } else {
              alt_feedback_text = [
                "The computer guessed the right color.",
                "The computer guessed the wrong color."
              ];
              html += '<div style="margin-bottom: 40px;" id="feedback-wrapper">'+alt_feedback_text[feedback_category]+'</div>';
            }
            html += '</div>'
            display_element.innerHTML = html;

            //end experiment
            setTimeout(function(){
              finish();
            }, feedbackDisplayDuration);
          }

          // compute all distances

          stimuli.forEach(function(listItem, index){
            loadJSON(listItem.stimulus, function(jsonFile) {
              distances.push(compareDistance(response.data, JSON.parse(jsonFile)));
              itemsProcessed++;
              if(itemsProcessed === stimuli.length) {
                changeFeedback("Analyzing signal ...", show_feedback());
              }
            });
          });

        } else {
          console.log('Compare only to true signal');


          function show_feedback() {

            // compute single distance
            var distance = compareDistance(response.data, response.stimulus);
            trial_data["feedback"] = distance;
            console.log("Distance", distance);

            if (trial.feedbackMode == 'learning') {
              // feedback during learning phase
              var feedback_category = scoreDistance(distance);
              $('#feedback-wrapper p').text(feedback_texts_learning[feedback_category]);

            } else if (trial.feedbackMode == 'single') {
              // feedback during test phase
              var success_probability = scoreProbability(distance);
              console.log("Success probability", success_probability);
              var feedback_category = Math.random() * 1 <= success_probability ? 0 : 2;
              if (feedback_category==0) { SCORE++; } //TODO: implement this via data.variable
              $('#feedback-wrapper p').text(feedback_texts_test[feedback_category]);

            } else {
              // ERROR: feedback mode unspecified
            }
          }

          changeFeedback("Analyzing signal ...", show_feedback);

          //end experiment
          setTimeout(function(){
            finish();
          }, feedbackDisplayDuration);
        }


      } else {
        console.log('No feedback given.');
        // mode = none => no feedback will be given
        finish();
      }

      function finish() {
        // clear the display
        display_element.innerHTML = '';

        // end trial
        jsPsych.finishTrial(trial_data);
      }

    }

  };

  return plugin;
})();
