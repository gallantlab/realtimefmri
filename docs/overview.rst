Overview
========

This package enables real-time collection and preprocessing of functional magnetic resonance imaging data along with stimulus presentation.

It consists of three main scripts that provide the following functionality:
 * `Data collection`_
 * `Preprocessing`_
 * `Stimulus presentation`_

And two smaller scripts to provide:
 * `Logging`_
 * `Synchronization with the scanner`_

_`Data collection` (``collect.py``)
-----------------------------------

``collect.py`` sits on the shared folder and continuously checks for new images coming off of the scanner. When an image appears, it is sent over a ``zmq`` messaging socket to the preprocessing script.

_`Preprocessing` (``preprocess.py``)
---------------------------------------

The data arriving from the data collection stage is formatted as a raw binary string of ``uint16`` values in the shape of a Siemens "mosaic" image. A few stages of preprocessing are needed to make it usable. The overall structure of preprocessing is simple: the ``Preprocess`` object sits on the `zmq` port and waits for image data to arrive. When it does it is passed through a series of preprocessing steps specified in the **preprocessing configuration file**. Each step adds its output to a dictionary of data holding references to all of the data objects that need to be passed between steps or sent on to the stimulation code. You can specify which of these data objects will be published to a ``zmq PUB`` socket, to which other code (such as the stimulation code) can subscribe.


_`Stimulus presentation` (``stimulate.py``)
---------------------------------------------

Things really get interesting when you making stimuli that depend on data gathered in real-time. All you need to do is build some software that manages a ``zmq SUB`` socket subscribed to one of the topics published by the preprocessing code. This can be implemented in any language that has has a `zmq` library, which is pretty much any language. Included in this package is a simple `pycortex <https://github.com/gallantlab/pycortex>`_ real-time brain activity visualizer.


_`Logging` (``logger.py``)
-------------------------

The main processes (collection, preprocessing, stimulation) are able to run on separate machines, which could have different clocks. To record all of these events to a single clock, each process can output log events over the network to a central logging process that saves a record of the run.

_`Synchronization with the scanner` (``sync.py``)
----------------------------------------------------
The scanner outputs a "5" keypress (via the FORP) to the **real-time computer** at the onset of each TR. This process captures those key presses and sends them off to the logger.