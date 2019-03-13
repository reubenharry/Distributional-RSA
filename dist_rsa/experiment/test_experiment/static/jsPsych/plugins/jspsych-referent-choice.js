/**
 * referent-choice
 * Matthias Hofer
 *
 * page that displays all (still availalbe) referents and allows
 * subject to choose which to relate to the computer
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["referent-choice"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "referent-choice",
    description: '',
    parameters: {
      text: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: undefined,
        description: 'The text to be displayed'
      },
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Referent',
        default: undefined,
        description: 'The correct referent to be seleted'
      },
      N_competitors: {
        type: jsPsych.plugins.parameterType.NUMERIC,
        pretty_name: 'Competitors',
        default: 2,
        description: 'How many competitor items'
      },
      complexity_class: {
        type: jsPsych.plugins.parameterType.NUMERIC,
        pretty_name: 'Complexity',
        default: 0,
        description: "What's the complexity of the item?"
      },
    }
  }


  plugin.trial = function(display_element, trial) {

    var html =  '';
    html += trial.text;

    var rIdx = 0;
    var competitor_set = []
    for (var i = 0; i < stimuli.length; i++) {
      // if (stimuli[i].complexity == trial.complexity_class && stimuli[i].referent != trial.stimulus) {
      if (stimuli[i].referent != trial.stimulus) {
        competitor_set.push(stimuli[i].referent);
      }
    }
    var choices = getRandomSubarray(competitor_set, trial.N_competitors);
    choices.push(trial.stimulus)
    choices = shuffle(choices);

    html += '<div class="referent-select-row">'
    for (var i = 0; i < choices.length; i++) {
      html += '<img id="'+rIdx+'" src="'+choices[i]+'" class="referent-choice-img">';
    }
    html += '</div>';
    display_element.innerHTML = html;

    var start_time = Date.now();
    // store response
    var response = {
      rt: null,
      selection: 0,
      correct: null,
      complexity_class: trial.complexity_class
    };

    // // change cursor for clickable elements
    $("img").css( 'cursor', 'pointer' );

    // // only allow still available referents to be clicked
    $("img").click(function(){

      response.selection = Number(this.id);
      if ($(this).attr("src")==trial.stimulus) {
        console.log('correct!');
        response.correct = 1;
      } else {
        console.log('incorrect!');
        response.correct = 0;
      }

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
        "selection": response.selection,
        "correct": response.correct
      };

      // clear the display
      display_element.innerHTML = '';

      // end trial
      jsPsych.finishTrial(trial_data);
    }

  };

  return plugin;
})();
