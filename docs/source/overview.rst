Overview
========

This package enables real-time collection, preprocessing, and display of functional magnetic resonance imaging data.

It consists of three main components that provide the following functionality:
 * `Data collection`_
 * `Preprocessing`_
 * `Dashboard`_
 * `Stimulus presentation`_

_`Data collection`
------------------

The first step is to load in the volumes as they are collected. It helps to understand the timeline for a single volume:

 1. The scanner sends a TTL pulse marking the time the volume is acquired.
 2. ``realtimefmri.collect_ttl`` receives the TTL pulse and stores a timestamp to a queue.
 3. The reconstruction computer finished reconstructing the volumes and saves it to the shared network drive.
 4. ``realtimefmri.collect_volumes`` detects a new file and stores its path to a queue.
 5. ``realtimefmri.collect`` merges the timestamps and volume by popping one entry from the timestamp queue and one from volume queue.
 6. The timestamped volume is sent off to preprocessing.

The volumes appear on a shared network drive as soon as the reconstruction process is complete. monitors a shared folder containing the images and loads them as they appear. It then sends over a ``zmq`` messaging socket to the preprocessing script.

_`Preprocessing`
----------------

The data arriving from the scanner are DICOM images of raw voxel values. A few stages of preprocessing are needed to make them usable. The overall structure of preprocessing is simple: the ``Preprocess`` object waits for image data to arrive. When it does it is passed through a series of preprocessing steps specified in the :ref:`preprocessing configuration file <pipelines>`. Each step adds its output to a dictionary of data holding references to all of the data objects that need to be passed between steps or sent to the dashboard or pycortex viewer for visualization.


_`Dashboard`
------------
The ``realtimefmri.stimulate.SendToDashboard`` pipeline step makes the data available for visualization in the dashboard. Several different plot types are available: bar plots, timeseries, images of array values. You can make many figures and configure which data are plotted in which figure.


_`Stimulus presentation`
------------------------

Things really get interesting when you making stimuli that depend on data gathered in real-time. All you need to do is build some software that reads from the redis database.