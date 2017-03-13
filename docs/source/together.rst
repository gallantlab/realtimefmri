Putting it all together
=======================

Real-time experiments have a lot of moving parts, so here's a step-by-step list of how to get one up and running.

Step-by-step
------------

1. Configure your :ref:`preprocessing and stimulation pipelines <pipelines>`. Names the files ``<preprocessing_pipeline_name>.yaml`` and ``<stimulus_pipeline_name>.yaml`` and add them to ``realtimefmri/pipelines``.

2. Connect the **real-time computer** to the :ref:`scanner network <network>`. You should be able to ``ls /mnt/scanner`` to view files on the **scanner console**. You'll need to a bit of detective work to find the ``<parent_directory>`` that contains the new runs.

3. Use ``scripts/realtime.py`` to run everything from one place. This script wraps  ``logging.py``, ``collect.py``, ``preprocess.py``, ``stimulate.py``, and ``sync.py`` for your convenience. Run it like this:


.. code-block:: bash

  cd realtimefmri/scripts
  python collect.py <recording_id> \
    <parent_directory> \
    <preprocessing_pipeline_name> \
    <stimulus_pipeline_name> \
    -p


(The ``-p`` is just a flag that says the provided ``<parent_directory>`` is a *parent* directory. It will look in that directory for the first new folder that is created and then monitor *that* folder for DICOM images.)

4. Log files containing the timing of all of the different aspects of the experiment are stored to ``realtimefmri/recordings/<recording_id>/recording.log``.

5. To exit the experiment, press ``Ctrl-C`` to send the interrupt signal.