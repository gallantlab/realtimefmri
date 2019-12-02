Overview
========

This package enables real-time collection, preprocessing, and display of functional magnetic resonance imaging data.


Data collection
---------------

The first step is to load in the volumes as they are collected. It helps to understand the timeline for a single volume:

 1. The scanner sends a TTL pulse marking the time the volume is acquired.
 2. ``realtimefmri.collect_ttl`` receives the TTL pulse and stores a timestamp to a queue.
 3. The reconstruction computer finished reconstructing the volumes and saves it to the shared network drive.
 4. ``realtimefmri.collect`` detects an incoming volume and merges it with a timestamp by popping one entry from the timestamp queue.
 5. The timestamped volume is sent off to preprocessing.


Preprocessing
-------------

The data arriving from the scanner are DICOM images of raw voxel values. A few stages of preprocessing are needed to make them usable. The overall structure of preprocessing is simple: the ``Preprocess`` object waits for image data to arrive. When it does it is passed through a series of preprocessing steps specified in the :ref:`preprocessing configuration file <pipelines>`. Each step adds its output to a dictionary of data holding references to all of the data objects that need to be passed between steps or sent to the dashboard or pycortex viewer for visualization.


Dashboard
---------
The ``realtimefmri.stimulate.SendToDashboard`` pipeline step makes the data available for visualization in the dashboard. Several different plot types are available: bar plots, timeseries, images of array values. You can make many figures and configure which data are plotted in which figure.


Stimulus presentation
---------------------

Things really get interesting when you making stimuli that depend on data gathered in real-time. All you need to do is build some software that reads from the redis database.
