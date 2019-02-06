/**
 * jspsych-image-keyboard-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/


jsPsych.plugins["svg-keyboard-response"] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'svg-keyboard-response',
    description: '',
    parameters: {
      cues: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Cues (4)',
        default: undefined,
        description: 'Cues to be used (need to be 4)'
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: 'Choices',
        default: jsPsych.ALL_KEYS,
        description: 'The keys the subject is allowed to press to respond to the stimulus.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the stimulus.'
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
        description: 'How long to show trial before it ends.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, trial will end when subject makes a response.'
      },
    }
  }

  plugin.trial = function(display_element, trial) {
    // create empty svg
    var SQUARE_SIZE = 300;
    var new_html = "<div id='svg-keyboard-response-stimulus'><svg id='svg' width=" + SQUARE_SIZE * 2 + " height=" + SQUARE_SIZE*2 + "></svg></div>";

    // add prompt
    if (trial.prompt !== null){
      new_html += trial.prompt;
    }

    // draw
    display_element.innerHTML = new_html;

    // change svg
    var s = Snap("#svg");
    var cue1 = s.rect(0,0,SQUARE_SIZE,SQUARE_SIZE);
    var cue2 = s.rect(SQUARE_SIZE,0,SQUARE_SIZE,SQUARE_SIZE);
    var cue3 = s.rect(0,SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE);
    var cue4 = s.rect(SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE);
    s.attr({
      fill: "white",
      stroke: "black",
      strokeWidth: 1,
    });
    var cues = ["cue1", "cue2", "cue3", "cue4"];
    var cues_rects = [cue1, cue2, cue3, cue4];
    for (i = 0; i < cues.length; i++) {
      this_cue = cues_rects[i];
      text_cue = trial.cues[i];
      this_cue_bb = this_cue.getBBox();
      x0 = this_cue_bb["x"] + this_cue_bb["w"]/2;
      y0 = this_cue_bb["y"] + this_cue_bb["h"]/2;
      var t = s.text(x0, y0, text_cue);
      t.attr({
        "text-anchor": "middle",
        fill: "#000000"
      });
    }
    // store response
    var response = {
      rt: null,
      key: null
    };

    // function to end trial when it is time
    var end_trial = function() {

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // kill keyboard listeners
      if (typeof keyboardListener !== 'undefined') {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
      }

      // gather the data to store for the trial
      var trial_data = {
        "rt": response.rt,
        "stimulus": trial.cues,
        "key_press": response.key
      };

      // clear the display
      s.clear();
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // function to handle responses by the subject
    var after_response = function(info) {

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      display_element.querySelector('#svg-keyboard-response-stimulus').className += ' responded';

      // only record the first response
      if (response.key == null) {
        response = info;
      }

      if (trial.response_ends_trial) {
        end_trial();
      }
    };

    // start the response listener
    if (trial.choices != jsPsych.NO_KEYS) {
      var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: false,
        allow_held_key: false
      });
    }

    // hide stimulus if stimulus_duration is set
    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#svg-keyboard-response-stimulus').style.visibility = 'hidden';
        s.clear();
      }, trial.stimulus_duration);
    }

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }

  };

  return plugin;
})();
