$(document).ready( function() {

	//TODO: storage and filename entry
	//TODO: function to activate and deactivate all buttons in document
	demoMode = true;

    var textFieldActive = false;

	var	visual_feedback_disabled = false,
		save_json_enabled = false,
		compare_input_enabled = false,
		b_color = ['#D3D3D3', '#4CAF50'];

	var outputFileName = "utterance";
	var inputFileName = "utterance-1";
	$( "#output-filename-field").val(outputFileName);
	$( "#input-filename-field").val(inputFileName);

    $( "#input-filename-field").keypress(function(e) {
        if(event.keyCode == 13){
            $('#reference_button').click();
        }
    });

    $( "input[type=text]" ).on("focus", function() {
      textFieldActive = true;
    });
    $( "input[type=text]" ).on("focusout", function() {
      textFieldActive = false;
    });

	//enable base play functionality
 	enablePlay();

 	// enable hotkeys
 	$(document).on('keydown', function(e) {
        if (textFieldActive) {
            return
        }

 		if(e.keyCode == 82 && !textFieldActive) { //record
 			$( "#record_button" ).click();
 		} else if (e.keyCode == 80 && !textFieldActive) { //playback
			$( "#playback_button" ).click();
		} else if (e.keyCode == 83 && !textFieldActive) { //save
 			$( "#save_button" ).click();
 		}
 	});

 	//record button
	$( "#record_button" ).click(function() {
		if (!record) {
			startRecording();
		} else if (record) {
			stopRecording();
			if (save_json_enabled) {
				fileName = $( "#output-filename-field").val();
				downloadJson(data_array, fileName + '.json');
			}

			if (compare_input_enabled) {
				console.log('Compare recorded utterance to input:');
				filePath = '/static/data/rec/' + $( "#input-filename-field").val() + '.json';
				loadJSON(filePath, function(jsonFile) {
			      	var reference_array = JSON.parse(jsonFile);
			      	distance = compareDistance(data_array, reference_array);
			      	console.log(distance);
			      	$( "#distance-value-field").val(Math.round(distance).toString());
			    });
			}
		}
		$( "#record_button" ).css('background-color', b_color[+record]);
	});

	//playback button
	$( "#playback_button" ).click(function() {
		if (record) { //if record is still enabled
			$( "#record_button" ).click();
		};
		if (data_array.length == 0) {
			alert("You must record an utterance first.");
			return false
		};

		// temporarily deactivate buttons
		$( "#record_button, #playback_button, #reference_button" ).disabled = true;
		$( "#playback_button" ).css('background-color', b_color[1]);

		//play back utterance here:
		playBackRecording(data_array, !visual_feedback_disabled, true, function() {
			$( "#record_button, #playback_button, #reference_button" ).disabled = false;
			$( "#playback_button" ).css('background-color', b_color[0]);
			enablePlay();
		});
	});

	$( "#save_button" ).click(function() {
		if (record) { //if record is still enabled
			$( "#record_button" ).click();
		}
		if (data_array.length == 0) {
			alert("You must record an utterance first.");
			return false
		};

		disablePlay();
		var tmp_text = $( "#save_button" ).html();
		$( "#save_button" ).disabled = true;
		$( "#save_button" ).html("Saving...");
		fileName = $( "#output-filename-field").val();
		downloadWav(data_array, fileName + '.wav').then(function() {
			$( "#save_button" ).html(tmp_text);
			$( "#save_button" ).disabled = false;
			enablePlay();
		});

	});

	$('#json_checkbox').change(function() {
        if(this.checked) {
            $(this).prop("checked", true);
            save_json_enabled = this.checked;
        } else {
        	$('#json_checkbox').val(this.checked);
        	save_json_enabled = this.checked;
	    }
    });

    $('#feedback_checkbox').change(function() {
        if(this.checked) {
            $(this).prop("checked", true);
            visual_feedback_disabled = this.checked;
        } else {
        	$('#feedback_checkbox').val(this.checked);
        	visual_feedback_disabled = this.checked;
	    }
    });

    $('#compare_checkbox').change(function() {
        if(this.checked) {
            $(this).prop("checked", true);
            compare_input_enabled = this.checked;
        } else {
        	$('#compare_checkbox').val(this.checked);
        	compare_input_enabled = this.checked;
	    }
    });

    //record button
	$( "#reference_button" ).click(function() {

		filePath = 'static/signals/' + $( "#input-filename-field").val() + '.json';
        console.log(filePath);
        $.get( filePath, function() {
            // other code here
        }).done(function() {

            // temporarily deactivate buttons
            $( "button" ).prop('disabled', true);
            $( "#reference_button" ).css('background-color', b_color[1]);

    		loadJSON(filePath, function(jsonFile) {
    	      	var utterance = JSON.parse(jsonFile);

    	      	playBackRecording(utterance, !visual_feedback_disabled, true, function() {
                    console.log('done!');
                    $( "button" ).prop('disabled', false);
    				$( "#reference_button" ).css('background-color', b_color[0]);
    				enablePlay();
    			});
    	    });

        }).fail(function() {
            alert( "File not found!" );
        })

	});
});
