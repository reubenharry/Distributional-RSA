/**
 * jspsych-flute-observation
 * Matthias Hofer
 *
 * plugin for displaying the flute (optional) and playing back an utterance
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["flute-observation"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "flute-observation",
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The utterance to be played'
      },
      visualFeedback: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Visual feedback',
        default: true,
        description: 'Whether or not to give visual feedback'
      },
      text: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: undefined,
        description: 'The text to be displayed'
      },
      returnToNeutral: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Return to neutral position',
        default: true,
        description: 'Whether or not the plunger returns at the end.'
      }
    }
  }


  plugin.trial = function(display_element, trial) {

    // display whistle
    var html =  ''
    html += trial.text;
    html += '<div id="flute-container"><div id="left_container">';
    html += '<div id="piston"></div><div id="body"></div></div>';
    html += '<div id="right_container"><div id="plunger"></div></div></div>';
    display_element.innerHTML = html;

    var start_time = Date.now();
    // store response
    var response = {
      rt: null,
      observed: [],
      feedback: trial.visualFeedback
    };

    //load json file and play
    loadJSON(trial.stimulus, function(jsonFile) {
      var utterance = JSON.parse(jsonFile);
      response.observed = utterance;
      playBackRecording(utterance, trial.visualFeedback, trial.returnToNeutral, end_trial);
    });

    function end_trial() {
      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      var end_time = Date.now();
      var rt = end_time - start_time;
      response.rt = rt;

      // gather the data to store for the trial
      var trial_data = {
        "rt": response.rt,
        "observed": response.observed,
        "visual_feedback": response.feedback
      };

      // clear the display
      display_element.innerHTML = '';

      // end trial
      jsPsych.finishTrial(trial_data);
    }
  };
  return plugin;
})();
