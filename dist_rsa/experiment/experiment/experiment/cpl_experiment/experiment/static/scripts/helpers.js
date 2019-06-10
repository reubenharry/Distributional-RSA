var DEBUG = true;

// https://github.com/eligrey/FileSaver.js
function saveAs(file, filename) {
    if (window.navigator.msSaveOrOpenBlob) // IE10+
        window.navigator.msSaveOrOpenBlob(file, filename);
    else { // Others
        var a = document.createElement("a"),
                url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
}

// function to skip a trial in jsPsych
var skip = function() {
  jsPsych.data.headphonecheck_passed = true
  jsPsych.finishTrial(jsPsych.data)
}
// add function that allows skipping trials by pressing 's'
$( document ).keypress(function(event) {
  if (event.which==115 & DEBUG) {
    skip();
  }
});

// DEBUG

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function argMin(array) {
  var invArray = array.map(function(x) { return x * (-1); });
  return argMax(invArray)
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function getRandomSubarray(arr, size) {
    var shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
}


function deep_copy(o) {
   var output, v, key;
   output = Array.isArray(o) ? [] : {};
   for (key in o) {
       v = o[key];
       output[key] = (typeof v === "object") ? deep_copy(v) : v;
   }
   return output;
}

function copy(o) {
  return JSON.parse(JSON.stringify(o))
}

async function audioBufferToWaveBlob(audioBuffer) {

  return new Promise(function(resolve, reject) {
    var worker = new Worker('scripts/waveWorker.js');
    worker.onmessage = function( e ) {
      var blob = new Blob([e.data.buffer], {type:"audio/wav"});
      resolve(blob);
    };

    let pcmArrays = [];
    for(let i = 0; i < audioBuffer.numberOfChannels; i++) {
      pcmArrays.push(audioBuffer.getChannelData(i));
    }

    worker.postMessage({
      pcmArrays,
      config: {sampleRate: audioBuffer.sampleRate}
    });
  });
}


function loadJSON(filename, callback) {
	var xobj = new XMLHttpRequest();
	xobj.overrideMimeType("application/json");
	xobj.open('GET', filename, true);
	xobj.onreadystatechange = function () {
		if (xobj.readyState == 4 && xobj.status == "200") {
			callback(xobj.responseText);
		}
	};
	xobj.send(null);
}

function cloneTimelineObject(timelineObject) {
  return JSON.parse(JSON.stringify( timelineObject ));
}

var assignChain = function(workerID, callback) {
  customURL = "/assign_chain?workerID=" + workerID
  $.ajax({
    dataType: "json",
    url: customURL,
    success: function(data) {
      callback(Number(data))
    }
  });
}

var releaseChain = function(chainID) {
  $.ajax({
    url: "/release_chain?chainID=" + chainID,
    success: function(data) {
      return data
    }
  });
}

// // Add bonus to task data
//   self.computeBonus = function(url, callback) {
//

var extendHIT = function() {
  $.ajax({
    url: "/extend_hit",
    success: function(data) {
      console.log(data);
    }
  });
}

var clearChaindDB = function() {
  // customURL = "/assign_chain?workerID=" + workerID
  // $.ajax({
  //   dataType: "json",
  //   url: customURL,
  //   success: function(data) {
  //     console.log("workerID: ", data);
  //     callback(Number(data))
  //   }
  // });
}
