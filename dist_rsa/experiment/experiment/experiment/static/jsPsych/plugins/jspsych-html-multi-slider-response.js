/**
 * jspsych-html-slider-response
 * a jspsych plugin for free response survey questions
 *
 * Josh de Leeuw
 *
 * documentation: docs.jspsych.org
 *
 */

var labels = ["not relevant", "relevant"]

jsPsych.plugins['html-multi-slider-response'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'html-multi-slider-response',
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The HTML string to be displayed'
      },
      min: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Min slider',
        default: 0,
        description: 'Sets the minimum value of the slider.'
      },
      max: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Max slider',
        default: 100,
        description: 'Sets the maximum value of the slider',
      },
      start: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Slider starting value',
        default: 50,
        description: 'Sets the starting value of the slider',
      },
      step: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Step',
        default: 1,
        description: 'Sets the step of the slider'
      },
      labels: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        pretty_name:'Labels',
        default: [],
        array: true,
        description: 'Labels of the slider.',
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        array: false,
        description: 'Label of the button to advance.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, trial will end when user makes a response.'
      },
    }
  }

  plugin.trial = function(display_element, trial) {

    var html = '<div id="jspsych-html-slider-response-wrapper" style="margin: 100px 0px; width=90%">';
    html += '<div id="jspsych-html-slider-response-stimulus">' + trial.stimulus + '</div>';
    html += '<div class="jspsych-html-slider-response-container" style="position:relative;">';
    html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-html-slider-response-response"></input>';
    html += '<div>'
    for(var j=0; j < trial.labels.length; j++){
      var width = 100/(trial.labels.length-1);
      var left_offset = j==0?(5 + (j * (100 /(trial.labels.length - 1))) - (width/2)):(-3 + (j * (100 /(trial.labels.length - 1))) - (width/2));
      // var left_offset = 10 + (j * (100 /(trial.labels.length - 1))) - (width/2);
      html += '<div style="display: inline-block; position: absolute; left:'+left_offset+'%; text-align: center; width: '+width+'%;">';
      html += '<span style="text-align: center; font-size: 80%;">'+labels[j]+'</span>';
      html += '</div>'
    }
    if (trial.prompt !== null){
      html += "<b>"+trial.prompt[0]+"</b>"+"<br>";
    }
    html += '</div>';
    html += '</div>';
    html += '</div>';


    html += '<div id="jspsych-html-slider-response-wrapper" style="margin: 100px 0px;">';
    // html += '<div id="jspsych-html-slider-response-stimulus">' + trial.stimulus + '</div>';
    html += '<div class="jspsych-html-slider-response-container" style="position:relative;">';
    html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-html-slider-response-response2"></input>';
    html += '<div>'
    for(var j=0; j < trial.labels.length; j++){
      var width = 100/(trial.labels.length-1);
      var left_offset = j==0?(5 + (j * (100 /(trial.labels.length - 1))) - (width/2)):(-3 + (j * (100 /(trial.labels.length - 1))) - (width/2));
      // var left_offset = 10 + (j * (100 /(trial.labels.length - 1))) - (width/2);
      html += '<div style="display: inline-block; position: absolute; left:'+left_offset+'%; text-align: center; width: '+width+'%;">';
      html += '<span style="text-align: center; font-size: 80%;">'+labels[j]+'</span>';
      html += '</div>'
    }
    html += "<b>"+trial.prompt[1]+"</b>"+"<br>";
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // if (trial.prompt !== null){

    html += '<div id="jspsych-html-slider-response-wrapper" style="margin: 100px 0px;">';
    // html += '<div id="jspsych-html-slider-response-stimulus">' + trial.stimulus + '</div>';
    html += '<div class="jspsych-html-slider-response-container" style="position:relative;">';
    html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-html-slider-response-response3"></input>';
    html += '<div>'
    for(var j=0; j < trial.labels.length; j++){
      var width = 100/(trial.labels.length-1);
      var left_offset = j==0?(5 + (j * (100 /(trial.labels.length - 1))) - (width/2)):(-3 + (j * (100 /(trial.labels.length - 1))) - (width/2));
      // var left_offset = 10 + (j * (100 /(trial.labels.length - 1))) - (width/2);
      html += '<div style="display: inline-block; position: absolute; left:'+left_offset+'%; text-align: center; width: '+width+'%;">';
      html += '<span style="text-align: center; font-size: 80%;">'+labels[j]+'</span>';
      html += '</div>'
    }
    html += "<b>"+trial.prompt[2]+"</b>"+"<br>";
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // if (trial.prompt !== null){

    html += '<div id="jspsych-html-slider-response-wrapper" style="margin: 100px 0px;">';
    // html += '<div id="jspsych-html-slider-response-stimulus">' + trial.stimulus + '</div>';
    html += '<div class="jspsych-html-slider-response-container" style="position:relative;">';
    html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-html-slider-response-response4"></input>';
    html += '<div>'
    for(var j=0; j < trial.labels.length; j++){
      var width = 100/(trial.labels.length-1);
      var left_offset = j==0?(5 + (j * (100 /(trial.labels.length - 1))) - (width/2)):(-3 + (j * (100 /(trial.labels.length - 1))) - (width/2));
      // var left_offset = 10 + (j * (100 /(trial.labels.length - 1))) - (width/2);
      html += '<div style="display: inline-block; position: absolute; left:'+left_offset+'%; text-align: center; width: '+width+'%;">';
      html += '<span style="text-align: center; font-size: 80%;">'+labels[j]+'</span>';
      html += '</div>'
    }
    html += "<b>"+trial.prompt[3]+"</b>"+"<br>";
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // if (trial.prompt !== null){
    // }

    // add submit button
    html += 'Please adjust all sliders before continuing.</br>';
    html += '<button id="jspsych-html-slider-response-next" class="jspsych-btn">'+trial.button_label+'</button>';



    display_element.innerHTML = html;

    var response = {
      rt: null,
      response: null
    };

    display_element.querySelector('#jspsych-html-slider-response-next').addEventListener('click', function() {
      // measure response time
      var endTime = (new Date()).getTime();
      response.rt = endTime - startTime;
      response.response = [
        trial.prompt[0],display_element.querySelector('#jspsych-html-slider-response-response').value,
        trial.prompt[1],display_element.querySelector('#jspsych-html-slider-response-response2').value,
        trial.prompt[2],display_element.querySelector('#jspsych-html-slider-response-response3').value,
        trial.prompt[3],display_element.querySelector('#jspsych-html-slider-response-response4').value,
        ];

      var cond1 = display_element.querySelector('#jspsych-html-slider-response-response').value!=50&&display_element.querySelector('#jspsych-html-slider-response-response2').value!=50&&display_element.querySelector('#jspsych-html-slider-response-response3').value!=50&&display_element.querySelector('#jspsych-html-slider-response-response4').value!=50


      if(cond1){
        end_trial();
      } else {
        var a =5
        // display_element.querySelector('#jspsych-html-slider-response-next').disabled = true;
      }

    });

    function end_trial(){

      jsPsych.pluginAPI.clearAllTimeouts();

      // save data
      var trialdata = {
        "rt": response.rt,
        "response": response.response,
        "stimulus": trial.stimulus
      };

      display_element.innerHTML = '';

      // next trial
      jsPsych.finishTrial(trialdata);
    }

    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#jspsych-html-slider-response-stimulus').style.visibility = 'hidden';
      }, trial.stimulus_duration);
    }

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }

    var startTime = (new Date()).getTime();
  };

  return plugin;
})();
