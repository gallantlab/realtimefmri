Putting it all together
=======================

Real-time experiments have a lot of moving parts, so here's a step-by-step list of how to get one up and running.

Step-by-step
------------

1. Configure your :ref:`preprocessing and stimulation pipelines <pipelines>`. Names the files ``<preprocessing_pipeline_name>.yaml`` and ``<stimulus_pipeline_name>.yaml`` and add them to ``.config/realtime/pipelines``.

2. Connect the **real-time computer** to the :ref:`scanner network <network>`. You should be able to ``ls /mnt/scanner`` to view files on the **scanner console**. You'll need to a bit of detective work to find the ``<parent_directory>`` that contains the new runs.

3. Check that the software can detect the TTL pulse from the scanner. The TTL comes in over USB. First, find which device file the USB is connected to using ``ls -l /dev/event/by-id``. Make sure that the ``~/.config/realtimefmri/config.cfg`` file has the correct file listed under ``[sync] keyboard = /dev/input/event#``. Finally make sure that the ``glab`` user has permissions to access that device. ``glab`` is in the ``plugdev`` group, so sure ``plugdev`` group has read and write access to the ``/dev/input/event##`` device (``sudo chown glab:plugdev /dev/input/event##``; ``sudo chmod g+rw /dev/input/event##``).

4. Launch the ``realtimefmri collect`` command to run everything from one place. This command wraps collection, synchronization, and logging for your convenience. Choose a unique <recording_id> for this run and enter the following command:


.. code-block:: bash

  realtimefmri collect <recording_id> \
    -v  # if you want verbose output


5. In another terminal, launch the ``realtime preprocess`` command to run preprocessing and stimulation pipelines.


.. code-block:: bash

  realtimefmri preprocess <recording_id> \
    <preprocessing_pipeline_name> \
    <stimulus_pipeline_name> \
    -v  # if you want verbose output


6. A log files containing event times is stored to ``~/.local/share/realtimefmri/recordings/<recording_id>/recording.log``.

7. To exit the experiment, press ``Ctrl-C`` to send the interrupt signal.


Simulating a real-time experiment outside of the scanner
--------------------------------------------------------

To test all of the moving parts without booking scanner time, we've provided a script that will simulate TTL pulses and volume acquisition.


.. code-block:: bash

    realtimefmri simulate <test_dataset>

Where ``test_dataset`` specifies a directory ``.local/share/realtimefmri/datasets/<test_dataset>`` containing ``.PixelData`` files. Be sure to specify ``/tmp/rtfmri`` as  ``<parent_directory_of_dicom_directory>`` in your call to ``realtimefmri console``. When the script starts, it will create a temporary folder within ``/tmp/rtfmri``. The console process will detect that new folder and start monitoring it for incoming volumes.

Start by pressing ``5`` to simulate the TTL pulse that occurs at the start of each volume acquisition. Then press enter twice to simulate the magnitude and phase volumes (only magnitude volumes are used).
