.. _network:

Connecting to the BIC scanner network
=====================================

The first thing you need to do connect to the BIC system so you can get the incoming brain volumes as they are collected. The BIC system consists of several computers operating on a local network:

 * **Scanner computer** controlling the scanner itself. This is attached to the magnet and is not for you to fiddle with.
 * **reconstruction computer**. Also not for you.
 * **scanner console**. This is the thing the operator interacts with. As soon as reconstruction is complete, the volume is saved to the hard drive on this computer.

You will be adding a fourth computer to the mix, the **real-time computer**, which will run the code in this package. This is accomplished by connecting it via ethernet to the router switch on top of the **scanner console**. This allows the **real-time computer** to access the image files as they appear in a shared directory on the *scanner console*.

Visit the `wiki <http://www/wiki/Real-time_fMRI>`_ for instructions on how to access images from the shared scanner network.
