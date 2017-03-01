Connecting to the BIC scanner network
=====================================

The first thing you'll have to do is get the incoming brain volumes hot off the scanner. The BIC system consists of several computers operating on a local network:

 * **Scanner computer** controlling the scanner itself. This is attached to the magnet and is not for you to fiddle with
 * **reconstruction computer**
 * **scanner console**

You will be adding a fourth computer to the mix, the **real-time computer**, which will run the code in this package. This is done by connecting it via ethernet to the router switch on top of the scanner computer. This allows the **real-time computer** to access the image files as they appear in a shared directory on the *scanner console*.

Visit the `wiki <http://www/wiki/Real-time_fMRI>`_ for instructions on how to access images from the shared scanner network.
