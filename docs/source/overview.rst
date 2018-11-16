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

The data arriving from the scanner are DICOM images of raw voxel values. A few stages of preprocessing are needed to make them usable. The overall structure of preprocessing is simple: the ``Preprocess`` object sits on the ``zmq`` port and waits for image data to arrive. When it does it is passed through a series of preprocessing steps specified in the :ref:`preprocessing configuration file <pipelines>`. Each step adds its output to a dictionary of data holding references to all of the data objects that need to be passed between steps or sent on to the stimulation code. You can specify which of these data objects will be published to a ``zmq PUB`` socket, to which other code (such as the stimulation code) can subscribe.


_`Dashboard`
------------


_`Stimulus presentation`
------------------------

Things really get interesting when you making stimuli that depend on data gathered in real-time. All you need to do is build some software that manages a ``zmq SUB`` socket subscribed to one of the topics published by the preprocessing code. This can be implemented in any language that has has a `zmq` library, which is pretty much any language. Included in this package is a simple `pycortex <https://github.com/gallantlab/pycortex>`_ real-time brain activity visualizer.