/**
 * jspsych-html-form
 * Matthias Hofer
 *
 * plugin for displaying the a form
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["html-form"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "html-form",
    description: '',
    parameters: {
      text: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Text',
        default: undefined,
        description: 'The text to be displayed'
      },
      feedback: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Feedback form',
        default: true,
        description: 'Whether or not to display the feedback form'
      },
      interface: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Feedback form',
        default: true,
        description: 'Whether or not to display the feedback form'
      },
    }
  }


  plugin.trial = function(display_element, trial) {

    // display whistle
    var html =  ''
    html += '<div id="form-header">' + trial.text + '</div>';

    // please answer a few questions before exiting

    html += '<div class="end-form-row">'
    html += '<label for="age">How old are you?</label>'
    html += '<select name="age" id="end_age">'
    html += '<option disabled selected value=""> -- select -- </option>'
    html += '</select>'
    html += '</div>'

    html += '<div class="end-form-row">'
    html += '<label for="gender">What is your gender?</label>'
    html += '<select id="end_gender" name="gender">'
    html += '<option disabled selected value=""> -- select -- </option>'
    html +=   '<option value="male">Male</option>'
    html +=   '<option value="female">Female</option>'
    html +=   '<option value="other">Other</option>'
    html += '</select>'
    html += '</div>'

    if (trial.interface) {
      html += '<div class="end-form-row">'
      html += '<label for="interface">Which interface did you use?</label>'
      html += '<select id="end_interface" name="interface">'
      html += '<option disabled selected value=""> -- select -- </option>'
      html +=   '<option value="mouse">Mouse</option>'
      html +=   '<option value="touchpad">Trackpad/Touchpad</option>'
      html +=   '<option value="stick">Pointing stick/Joystick</option>'
      html +=   '<option value="other">Other</option>'
      html += '</select>'
      html += '</div>'
    }

    html += '<div class="end-form-row">'
    html += '<label for="pitch">Do you have absolute/perfect pitch?</label>'
    html += '<select id="end_pitch" name="pitch">'
    html += '<option disabled selected value=""> -- select -- </option>'
    html +=   '<option value="yes">Yes</option>'
    html +=   '<option value="no">No</option>'
    html +=   '<option value="unsure">Don\'t know</option>'
    html += '</select>'
    html += '</div>'

    html += '<div class="end-form-row">'
    html += '<label for="music">Do you play a musical instrument?</label>'
    html += '<select id="end_music" name="music">'
    html += '<option disabled selected value=""> -- select -- </option>'
    html +=   '<option value="yes">Yes</option>'
    html +=   '<option value="no">No</option>'
    html += '</select>'
    html += '</div>'

    if (trial.feedback) {
      html += '<div class="end-form-row">'
      html += '<label for="feedback">Please share any general feedback about the experiment that would help us improve your experience.</label>'
      html += '<textarea name="feedback" id="feedback_text_field" rows="6" cols="43"></textarea>'
      html += '</div>'
    }

    html += '<div class="end-form-row">'
    html += '<button type="button" class="btn btn-primary btn-lg" id="end_experiment_button">End experiment</button>'
    html += '</div>'

    // languages besides english
    display_element.innerHTML = html;


    var start_time = Date.now();
    // store response

    var response = {
      rt: null,
      age: '',
      gender: '',
      interface: '',
      pitch: '',
      music: '',
      feedback: ''
    };


    // form functionality using jquery
    $(function() {
      var $select = $("#end_age");
      for (i=18; i<=99; i++) {
        $select.append($('<option></option>').val(i).html(i));
      }
    });

    $( "#end_experiment_button" ).click(function() {

      response['age'] = $("#end_age option:selected").val();
      response['gender'] = $("#end_gender option:selected").val();

      response['pitch'] = $("#end_pitch option:selected").val();
      response['music'] = $("#end_music option:selected").val();

      if (trial.feedback) {
        response['feedback'] = $("#feedback_text_field").val();
      } else {
        response['feedback'] = 'no response';
      }

      if (trial.interface) {
        response['interface'] = $("#end_interface option:selected").val();
      } else {
        response['interface'] = 'no response';
      }


      if (response['age']!='' && response['gender']!='' &&
        response['interface']!='' && response['pitch']!='' &&
        response['music']!='' && response['feedback']!=''){
        end_trial();
      } else {
        alert("Please respond to each item on the form.");
      }
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
        "age": response.age,
        "gender": response.gender,
        "interface": response.interface,
        "pitch": response.pitch,
        "music": response.music,
        "feedback":response.feedback
      };

      // clear the display
      display_element.innerHTML = '';

      // end trial
      jsPsych.finishTrial(trial_data);
    }
  };
  return plugin;
})();
