.. _network:

Acquiring DICOM images in real-time
===================================

To run a real-time experiment you need to be able to access the volumes as they are acquired. This involves connecting the real-time computer to the scanner console, and configuring the scanner to write out DICOM files in real-time.

These steps should all be performed **before registing the patient for scanning** or DICOM files will not be created in real-time.

Connecting to the scanner network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Connect the **real-time computer** to the router switch on top of the **scanner console** with an ethernet cable.
2. Take note of the real-time computer's IP address, which should be ``192.168.1.<some number>``.

Mounting the network shared drive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The real-time computer hosts a network shared folder using the `SAMBA <https://www.samba.org/>`_ protocol. We will use the built-in Windows tool to mount the shared folder over the network.

1. Press Ctrl+Esc to open the Windows start menu, click "Computer" or "Windows Explorer."
2. Click "Tools > Map network drive", and enter ``Y:`` and ``\\<real-time computer's IP address>\rtfmri``
3. Check "use different credentials" and click "OK."
4. Enter ``rtfmri`` for the user name and enter the samba password and click "OK."

This links the directory ``Y:`` on the scanner console to the ``/mnt/scanner`` on the ``realtimefmri_samba`` container running on the real-time computer. You can test this connection by writing a file to the directory on one computer and making sure it appears in the corresponding directory on the other computer.


Configuring the scanner to write out DICOM files to that folder in real-time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Siemens scanners come with a useful program called ``ideacmdtool`` that can configure the scanner to write out DICOM files as they are collected. Unfortunately it is not that well documented -- thanks to the `FIRMM <http://firmm.readthedocs.io/en/latest/>`_ and `Dynamically Adaptive Imaging <http://imaging.mrc-cbu.cam.ac.uk/basewiki/DynamicallyAdaptiveImaging>`_ projects for explaining how to configure it.

1. Press Ctrl-Esc to open the Windows start menu. Click on "Advanced user" to activate Advanced User mode. Talk to scanner administrators to get the authentication details.
2. Press Ctrl-Esc and click "Command Prompt" or click "Run" and enter ``cmd`` in the dialog to launch the command prompt.
3. Enter ``ideacmdtool`` in the command prompt to start the ``ideacmdtool`` program.

Configure the default settings:

4. Enter ``4`` to go to the "Online export defaults" menu
5. Enter ``1`` and set "Target port" to ``-1``
6. Enter ``3`` and set "Target path" to ``y:``
7. Enter ``5`` to set ``SendBuffered`` to ``OFF``
8. Enter ``q`` to go back to the main ``ideacmdtool`` menu

Configure the session settings:

9. Enter ``5`` to go to the "Switches" menu
10. Enter ``8`` to set ``SendIMA`` to ``ON``
11. Enter ``q`` to go back to the main ``ideacmdtool`` menu
12. Enter ``q`` to exit ``ideacmdtool``
