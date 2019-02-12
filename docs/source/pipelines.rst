.. _pipelines:

Pipelines
=========

Part of flexibility of this package comes from the ability to configure custom preprocessing pipelines. A pipeline describes a sequence of processing steps that each new sample undergoes. Pipelines are specified using :mod:`yaml` files stored in the ``pipelines`` directory.


Configuring a real-time preprocessing pipeline
----------------------------------------------

A pipeline must be able to pass data between the different steps. This is accomplished using a dictionary of objects, the ``data_dict``. Each step receives its input from the ``data_dict`` and adds its output to the ``data_dict`` for subsequent steps to use.

The basic format of a preprocessing pipeline configuration file is:

.. code-block:: yaml

  # arguments sent to all preprocessing steps (steps will ignore keywords that
  # they do not use, so put anything that you want to re-use across many steps.
  # pycortex subject and transform names are used often)
  global_defaults:
    subject: S1
    xfm_name: 20150101S1_transform

  # a list of steps in the pipeline
  pipeline:
    - name: name_of_step
      # the name of a python class that subclasses PreprocessingStep
      class_name: realtimefmri.preprocess.StepClass

      # keyword arguments used to initialize the step
      kwargs:
        a_keyword_argument: 5324532
        another_one: wefwef

      # keys indicating which data_dict values this step processeses
      input:
        - input_key

      # return values will be added to data_dict with this key
      output:
        - output_key



Example pipeline
----------------

Here's an example of a complete pipeline configuration. It converts the raw DICOM to nifti, motion corrects, extracts the motion parameters from the motion correction affine, extracts the gray matter voxels, computes their mean and standard deviations, computes their z-score, decodes using a pre-trained decoder, sends an image of the decoded category to the :ref:`dashboard <dashboard>`, sends the motion parameters to the dashboard, and sends the gray matter activity to the pycortex viewer.

.. code-block:: yaml

  global_parameters:
    n_skip: 0

  pipeline:
    - name: motion_correct
      class_name: realtimefmri.preprocess.MotionCorrect
      kwargs: { output_transform: True }
      input: [ raw_image_nii ]
      output: [ nii_mc, affine_mc ]

    - name: decompose_affine
      class_name: realtimefmri.preprocess.Function
      kwargs : { function_name: realtimefmri.image_utils.decompose_affine }
      input: [ affine_mc ]
      output: [ pitch, roll, yaw, x_displacement, y_displacement, z_displacement ]

    - name: nifti_to_volume
      class_name: realtimefmri.preprocess.NiftiToVolume
      input: [ nii_mc ]
      output: [ volume ]

    - name: gm_mask
      class_name: realtimefmri.preprocess.ApplyMask
      kwargs: { mask_type: thick }
      input: [ volume ]
      output: [ gm_responses ]

    - name: incremental_mean_std
      class_name: realtimefmri.preprocess.IncrementalMeanStd
      input: [ gm_responses ]
      output: [ gm_mean, gm_std ]

    - name: zscore
      class_name: realtimefmri.preprocess.ZScore
      input: [ gm_responses, gm_mean, gm_std ]
      output: [ gm_zscore ]

    - name: decode
      class_name: realtimefmri.preprocess.SklearnPredictor
      kwargs: { surface: TZ, pickled_predictor: TZ_motor_decoder_thick.pkl }
      input: [ gm_zscore ]
      output: [ prediction ]

    - name: select_predicted_image
      class_name: realtimefmri.preprocess.Dictionary
      kwargs: { dictionary: { hand: static/img/motor_decoder/hand.png,
                              foot: static/img/motor_decoder/foot.png,
                              mouth: static/img/motor_decoder/mouth.png,
                              saccade: static/img/motor_decoder/saccade.png,
                              speak: static/img/motor_decoder/speak.png },
                decode_key: utf-8 }
      input: [ prediction ]
      output: [ image_url ]

    - name: send_motion_parameters
      class_name : realtimefmri.stimulate.SendToDashboard
      kwargs: { name: motion_parameters, plot_type: timeseries }
      input: [ pitch, roll, yaw ]

    - name: send_motion_parameters_x
      class_name : realtimefmri.stimulate.SendToDashboard
      kwargs: { name: x_disp, plot_type: timeseries }
      input: [ x_displacement ]

    - name: send_motion_parameters_y
      class_name : realtimefmri.stimulate.SendToDashboard
      kwargs: { name: y_disp, plot_type: timeseries }
      input: [ y_displacement ]

    - name: send_motion_parameters_z
      class_name : realtimefmri.stimulate.SendToDashboard
      kwargs: { name: z_disp, plot_type: timeseries }
      input: [ z_displacement ]

    - name: send_prediction
      class_name : realtimefmri.stimulate.SendToDashboard
      kwargs: { name: predicted_image, plot_type: static_image }
      input: [ image_url ]

    - name: flatmap
      class_name: realtimefmri.stimulate.SendToPycortexViewer
      kwargs: { name: flatmap }
      input: [ gm_zscore ]
