.. _pipelines:

Configuring a real-time pipeline
=====================================

The flexibility of this package comes from your ability to configure custom preprocessing and stimulation pipelines. To run a real-time experiment, you'll need to configure these. Pipelines are specified using :mod:`yaml` files stored in the ``pipelines`` directory.


Preprocessing pipeline
----------------------
A pipeline must be able to pass data between the different steps. This is accomplished using a dictionary of objects, the ``data_dict``. Each step receives its input from the ``data_dict`` and adds its output to the ``data_dict`` for subsequent steps to use.

The basic format of a preprocessing pipeline configuration file is:

.. code-block:: yaml

  # arguments sent to all preprocessing steps (steps will ignore keywords that
  # they do not use, so put anything that you want to reuse across many steps.
  # subject name and transform name are common)
  global_defaults:
    subject: S1
    xfm_name: 20150101S1_transform

  # a list of steps in the pipeline
  pipeline:
    - name: name_of_step
      # yaml tag specifying a python object (one that subclasses
      # PreprocessingStep)
      instance: !!python/object:realtimefmri.preprocess.StepClass {}

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

      # data are sent over a PUB-SUB socket. this structure uses a topic to 
      # address data to the right subscribers. indicate a topic here that can
      # be received by the stimulation script
      send:
        - publish_to_topic


Stimulation pipeline
--------------------
Stimulation pipeline is configured similarly to preprocessing pipeline, except inputs are specified by mapping the topic names (that were published to in the preprocessing script) to variable names used by the stimulation steps internally.
  
.. code-block:: yaml

  global_defaults:
    subject: &SUBJECT S1

  # components that do not receive input from preprocessing and simply run as
  # soon as the pipeline commences
  initialization:
    - name: a_name 
      instance: !!python/object:realtimefmri.stimulate.SomeClass {}
      kwargs:
        keyword_argument_1: hi_there
        keyword_argument_2: 123

  # these components receive inputs sent over from the preprocessing pipeline
  pipeline:
    - name: responsive_stimulus
      instance: !!python/object:realtimefmri.stimulate.AnotherClass {}
      # key: topic that the preprocessing script published to
      # value: the name the data gets inside of the stimulation .run() class
      topic: { publish_to_topic: variable_name }
      kwargs:
        some_kwargs: *SUBJECT
        another_kwargs: 234234


Complete example
----------------

Here's an example of a complete pipeline configuration. It converts the raw
binary to nifti, motion corrects, saves a copy of the motion corrected nifti, extracts the gray matter voxels, computes their mean and standard deviations, computes their z-score, and sends that result off to the stimulation code. Pay attention to the ``input`` and ``output`` keys. Those are the keys to
the ``data_dict`` that show you how the data is transformed as it passes
through the pipeline.

Here is the preprocessing configuration. The first step in the pipeline should be a ``RawToNifti`` object. It receives the raw binary mosaic sent over from the data collection script and turns it into a nifti file. The ``input`` should always be ``[raw_image_binary]``, which is the name given to the image that arrives over the ``zmq`` socket from the data collection script. The pipeline proceeds from here, passing the volume through a series of steps.

.. code-block:: yaml

  global_defaults:
    subject: &SUBJECT S1
    xfm_name: &XFMNAME 20150101S1_transform
    nskip: 5

  pipeline:
    - name: raw_to_nifti
      instance: !!python/object:realtimefmri.preprocess.RawToNifti {}
      input: [ raw_image_binary ]
      output: [ image_nifti ]

    - name: motion_correct
      instance: !!python/object:realtimefmri.preprocess.MotionCorrect {}
      input: [ image_nifti ]
      output: [ image_nifti_mc ]
    
    - name: save_nifti
      instance: !!python/object:realtimefmri.preprocess.SaveNifti {}
      input: [ image_nifti_mc ]

    - name: extract_gm_mask
      instance: !!python/object:realtimefmri.preprocess.ApplyMask {}
      kwargs: { mask_type: thick }
      input: [ image_nifti_mc ]
      output: [ gm_activity ]

    - name: running_mean_std
      instance: !!python/object:realtimefmri.preprocess.OnlineMoments {}
      input: [ gm_activity ]
      output:
        - gm_activity_mean
        - gm_activity_std

    - name: gm_activity_zscore
      instance: !!python/object:realtimefmri.preprocess.VoxelZScore {}
      input:
        - gm_activity
        - gm_activity_mean
        - gm_activity_std
      output: [ gm_activity_zscore ]
      send: [ gm_activity_zscore ]

And the stimulation configuration. This example launches a ``pycortex`` viewer that will display brain activity in real-time. As you can see, it only has access to the 
topics that the preprocessing pipeline publishes, i.e., ``gm_activity_zscore``.

.. code-block:: yaml

  global_defaults:
    subject: &SUBJECT S1

  initialization:
    - name: record_microphone_input 
      instance: !!python/object:realtimefmri.stimulate.AudioRecorder {}
      kwargs:
        jack_port: "system:capture_1"
        file_name: microphone

  pipeline:
    - name: pycortex_viewer
      instance: !!python/object:realtimefmri.stimulate.PyCortexViewer {}
      topic: { gm_activity_zscore: data }
      kwargs:
        subject: *SUBJECT
        xfm_name: 20150101S1_transform
        mask_type: thick
        vmin: -0.01
        vmax: 0.01

    - name: debug
      instance: !!python/object:realtimefmri.stimulate.Debug {}
      topic: { gm_activity_zscore: data }
