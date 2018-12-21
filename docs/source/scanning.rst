Scanning
========

Real-time experiments have a lot of moving parts, so here's a step-by-step list of how to get one up and running.


Prior to scanning
-----------------

1. Configure a :ref:`preprocessing pipeline <pipelines>`.


During the scanning session
---------------------------

2. Follow the instructions for how to :ref:`acquire DICOM images in real-time<network>`.
3. Configure the software to listen for TTLs on the correct device. Find which device the USB is connected to using ``ls -l /dev/event/by-id``. Set ``EVENT_DEVICE=/dev/input/event<device number>`` in the ``.env`` file.
4. Run ``make docker.up`` to start the real-time fMRI software.


Control panel
^^^^^^^^^^^^^
5. Go to http://localhost:8050/controls in your web browser.
6. Select "keyboard" as the "TTL source..." and start the data collectors by clicking the ▶ buttons next to "Collect TTL", "Collect volumes", and "Collect" (in that order).
7. Select the Preprocessing pipeline using the "Configuration..." dropdown.
8. Select the pycortex surface, transform, and mask using the dropdowns.
9. Start the preprocessor by clicking the ▶ button next to "Preprocess".
10. If using the pycortex webgl viewer, click the ▶ button next to "Viewer" and go to http://localhost:8051 in another web browser window.


Dashboard
^^^^^^^^^
11. Go to http://localhost:8050/controls in your web browser.
12. Click the ↺ button to refresh the available data sources.
13. Click the + button to add a graph. A dropdown menu will appear for each new graph added.
14. Select the data sources for each graph using the dropdown menus.
