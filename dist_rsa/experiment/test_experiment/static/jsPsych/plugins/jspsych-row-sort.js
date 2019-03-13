/**
 * jspsych-row-sort
 * plugin for drag-and-drop sorting of a collection of images
 * Josh de Leeuw
 *
 * documentation: docs.jspsych.org
 */


jsPsych.plugins['row-sort'] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('row-sort', 'stimuli', 'image');

  plugin.info = {
    name: 'row-sort',
    description: '',
    parameters: {
      stimuli: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Stimuli',
        default: undefined,
        array: true,
        description: 'Images to be displayed.'
      },
      stim_height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus height',
        default: 100,
        description: 'Height of images in pixels.'
      },
      stim_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus width',
        default: 100,
        description: 'Width of images in pixels'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'It can be used to provide a reminder about the action the subject is supposed to take.'
      },
      left_text: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Left text',
        default: null,
        description: 'Text to be displayed on the left hand side.'
      },
      right_text: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Right text',
        default: null,
        description: 'Text to be displayed on the right hand side.'
      },
      prompt_location: {
        type: jsPsych.plugins.parameterType.SELECT,
        pretty_name: 'Prompt location',
        options: ['above','below'],
        default: 'above',
        description: 'Indicates whether to show prompt "above" or "below" the sorting area.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'The text that appears on the button to continue to the next trial.'
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    var start_time = (new Date()).getTime();

    var html = "";
    // check if there is a prompt and if it is shown above
    if (trial.prompt !== null && trial.prompt_location == "above") {
      html += trial.prompt;
    }

    html += '<div '+
      'id="jspsych-row-sort-arena" '+
      'class="jspsych-row-sort-arena" '+
      '><ul id="sortable"></ul><div id="row-text-description">'+
      '<p style="float: left; text-align: left; width: 50%;">'+trial.left_text+'</p>' +
      '<p style="float: left; text-align: right; width: 50%;">'+trial.right_text+'</p>' +
      '</div></div>';

    // check if prompt exists and if it is shown below
    if (trial.prompt !== null && trial.prompt_location == "below") {
      html += trial.prompt;
    }

    display_element.innerHTML = html;

    // randomize array
    var rand_order = shuffle([...Array(trial.stimuli.length).keys()]);

    for (var i = 0; i < trial.stimuli.length; i++) {
      j = rand_order[i]

      stim_values = trial.stimuli[j].split('/').pop().slice(0, -4).split('-');
      signal_id = stim_values[5];
      complexity = stim_values[7];

      display_element.querySelector("#sortable").innerHTML += '<li class="ui-state-default"><img '+
        'src="'+trial.stimuli[j]+'" data-id="'+signal_id+'" data-complexity="'+complexity+'" class="jspsych-row-sort-sortable"></li>'
    }

    var initial_order = [];
    var initial_complexity = [];
    var matches = display_element.querySelectorAll('.jspsych-row-sort-sortable');
    for(var i=0; i<matches.length; i++){
      initial_order.push(matches[i].dataset.id);
      initial_complexity.push(matches[i].dataset.complexity);
    }

    display_element.innerHTML += '<button id="jspsych-row-sort-done-btn" class="jspsych-btn">'+trial.button_label+'</button>';

    $(function() {
      $( "#sortable" ).sortable({
        placeholder: "ui-state-highlight",
        start: function(e, ui) {
          ui.placeholder.width(ui.item.width());
          ui.placeholder.height(ui.item.height());
        }
      });
      $( "#sortable" ).disableSelection();
    });

    display_element.querySelector('#jspsych-row-sort-done-btn').addEventListener('click', function(){

      var end_time = (new Date()).getTime();
      var rt = end_time - start_time;
      // gather data
      // get final position of all objects
      var final_order = [];
      var final_complexity = [];
      var matches = display_element.querySelectorAll('.jspsych-row-sort-sortable');
      for(var i=0; i<matches.length; i++){
        final_order.push(matches[i].dataset.id);
        final_complexity.push(matches[i].dataset.complexity);
      }

      var trial_data = {
        "initial_order": initial_order,
        "final_order": final_order,
        "initial_complexity": initial_complexity,
        "final_complexity": final_complexity,
        "rt": rt
      };

      // advance to next part
      display_element.innerHTML = '';
      jsPsych.finishTrial(trial_data);
    });

  };

  return plugin;
})();
