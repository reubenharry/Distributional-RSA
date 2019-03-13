/**
 * select-referent-page
 * Matthias Hofer
 *
 * page that displays all (still availalbe) referents and allows
 * subject to choose which to relate to the computer
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["select-referent-page"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "select-referent-page",
    description: '',
    parameters: {
      text: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: undefined,
        description: 'The text to be displayed'
      },
    }
  }


  plugin.trial = function(display_element, trial) {


    var html =  '';
    html += trial.text;

    nReferents = stimuli.length;
    nRows = Math.floor(Math.sqrt(nReferents));
    nCols = Math.ceil(nReferents/nRows);

    // build referent matrix
    var rIdx = 0,
        classTag = '';
    for (var i = 0; i < nRows; i++) {
      var rowHTML = '';
      for (var j = 0; j < nCols; j++) {
        if (rIdx <= stimuli.length - 1) { // if there are still referents to choose from
          classTag = (TESTED_IDs.indexOf(rIdx)==-1) ? ' available' : ' unavailable';
          rowHTML += '<img id="'+rIdx+'" src="'+stimuli[rIdx].referent+'" class="select-referent-img'+classTag+'">';
          rIdx++;
        }
      }
      html += '<div class="referent-select-row">' + rowHTML;
      html += '</div>';
    }

    display_element.innerHTML = html;

    var start_time = Date.now();
    // store response
    var response = {
      rt: null,
      selection: 0
    };

    // change cursor for availalbe referents
    $(".available").css( 'cursor', 'pointer' );

    // only allow still available referents to be clicked
    $(".available").click(function(){
      ref_ID = Number(this.id)
      response.selection = ref_ID;
      CURRENT_STIMULUS_ID = ref_ID;
      TESTED_IDs.push(CURRENT_STIMULUS_ID);
      end_trial();
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
        "selection": response.selection
      };

      // clear the display
      display_element.innerHTML = '';

      // end trial
      jsPsych.finishTrial(trial_data);
    }

  };

  return plugin;
})();
