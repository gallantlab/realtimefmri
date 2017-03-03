Preprocessing
=============

The following steps can be combined to produce a preprocessing pipeline. They are all classes that subtype ``PreprocessingStep``.


Modules
-------

  * `RawToNifti`_
  * `SaveNifti`_
  * `MotionCorrect`_
  * `ApplyMask`_
  * `ApplyMask2`_
  * `ActivityRatio`_
  * `RoiActivity`_
  * `WMDetrend`_
  * `OnlineMoments`_
  * `RunningMeanStd`_
  * `VoxelZscore`_



_`RawToNifti`:
++++++++++++++

This should be the first step in the pipeline. It receives the raw binary mosaic sent over from the data collection script and turns it into a nifti file. The ``input`` should always be ``[raw_image_binary]``, which is the name given to the image that arrives over the ``zmq`` socket from the data collection script.

Example:

.. code-block:: yaml

  pipeline:
  - name: raw_to_nifti
    instance: !!python/object:realtimefmri.preprocessing.RawToNifti {}
    input: [ raw_image_binary ]
    output: [ image_nifti ]




_`SaveNifti`
++++++++++++

Saves its input nifti file to the ``<recording_id>/nifti`` directory. ``<recording_id>`` will usually be provided upon initialization of the ``Preprocessor`` object. (If omitted, the folder name will be the subject and date.)

Example:

.. code-block:: yaml

  pipeline:
    - name: save_nifti
      instance: !!python/object:realtimefmri.preprocessing.SaveNifti {}
      input: [ image_nifti_mc ]

_`MotionCorrect`
++++++++++++++++

Example:

.. code-block:: yaml

  pipeline:
    - name: motion_correct
      instance: !!python/object:realtimefmri.preprocessing.MotionCorrect {}
      input: [ image_nifti ]
      output: [ image_nifti_mc ]


_`ApplyMask`
++++++++++++

Apply a ``pycortex`` mask to the image volume. Must supply the ``subject``, ``xfm_name``, and ``mask_type`` as keyword arguments.

Example:

Apply the thick gray matter mask

.. code-block:: yaml

  pipeline:
    - name: extract_gm_mask
      instance: !!python/object:realtimefmri.preprocessing.ApplyMask {}
      kwargs:
        subject: S1
        xfm_name: 20150101S1_transform
        mask_type: thick
      input: [ image_nifti ]
      output: [ gm_activity ]


_`ApplyMask2`
+++++++++++++

Example:

.. code-block:: yaml

  pipeline:


_`ActivityRatio`
++++++++++++++++

Example:

.. code-block:: yaml

  pipeline:


_`RoiActivity`
++++++++++++++

Example:

.. code-block:: yaml

  pipeline:


_`WMDetrend`
++++++++++++

Example:

.. code-block:: yaml

  pipeline:


_`OnlineMoments`
++++++++++++++++

Example:

.. code-block:: yaml

  pipeline:
    - name: running_mean_std
      instance: !!python/object:realtimefmri.preprocessing.OnlineMoments {}
      input: [ gm_activity ]
      output:
        - gm_activity_mean
        - gm_activity_std


_`RunningMeanStd`
+++++++++++++++++

Example:

.. code-block:: yaml

  pipeline:


_`VoxelZscore`
++++++++++++++

Example:

.. code-block:: yaml
  
  pipeline:
  
    - name: gm_activity_zscore
      instance: !!python/object:realtimefmri.preprocessing.VoxelZScore {}
      input:
        - gm_activity
        - gm_activity_mean
        - gm_activity_std
      output: [ gm_activity_zscore ]
      send: [ gm_activity_zscore ]
