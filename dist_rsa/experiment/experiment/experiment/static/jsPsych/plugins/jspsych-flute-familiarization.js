/**
 * jspsych-flute-familiarization
 * Matthias Hofer
 *
 * plugin for displaying the flute (optional) and playing back an utterance
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["flute-familiarization"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "flute-familiarization",
    description: '',
    parameters: {
      text: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: undefined,
        description: 'The text to be displayed'
      },
      headerText: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: '',
        description: 'The text to be displayed'
      },
      promptText: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: 'Please familiarize yoursef with the instrument!',
        description: 'The text to be displayed'
      },
    }
  }


  plugin.trial = function(display_element, trial) {

    inactivate = false;
    // display whistle
    var html =  ''
    html += '<div id="instruction-wrapper"><div id="header-text" class="warning">'+ trial.headerText  +'</div><div style="display: inline-block;">'
    html += '<div id="instruction-box">'+ trial.text  +'</div>'
    html += '<div id="flute-container" style="border-width: 2px; float: left !important;"><div id="left_container">';
    html += '<div id="piston"></div><div id="body"></div></div>';
    html += '<div id="right_container"><div id="plunger"></div></div></div></div></div>';
    html += '<div id="promptText" style="display: none;">'+ trial.promptText+ '</div>';

    display_element.innerHTML = html;

    var start_time = Date.now();

    // store response
    var response = {
      rt: null,
      data: []
    };

    $(document).ready(function() {
      enablePlay(end_trial);
      startRecording();
    });

    function end_trial() {
      inactivate = true;

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();
      response.data = data_array;

      var end_time = Date.now();
      var rt = end_time - start_time;
      response.rt = rt;

      // gather the data to store for the trial
      var trial_data = {
        "rt": response.rt,
        "data" : response.data
      };

      // clear the display
      display_element.innerHTML = '';

      // end trial
      jsPsych.finishTrial(trial_data);
    }
  };
  return plugin;
})();
