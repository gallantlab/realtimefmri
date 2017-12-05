.. _network:

Connecting to the scanner network
=================================

To run a real-time experiment you need to be able to access the volumes as they are acquired. This starts with connecting to the BIC network.

Network layout
--------------
The BIC consists of several computers operating on a local network:
 * **Scanner computer** controlling the scanner itself. This is attached to the magnet and is not for you to fiddle with.
 * **Reconstruction computer**. Makies an image from all those measurements the scanner collects. Also not for you.
 * **Scanner console**. The computer the scanner operator interacts with to run a scan. All of the tweaks take place on this computer.

You will be adding a fourth computer to the mix, the **real-time computer**, which will run the code in this package. This is accomplished by connecting it via ethernet to the router switch on top of the **scanner console**. This allows the **scanner console** to write images into a folder that is shared with the **real-time computer**.


Setting up Samba share for the first time
-----------------------------------------

First, configure Samba to share a directory with the **real-time computer**. This only needs to be done once. Install ``samba`` package:

.. code-block:: bash
    
    sudo apt-get update
    sudo apt-get install samba


Samba manages its own passwords (i.e., they are not the same as your Linux passwords), so create a new samba password for your user with:

.. code-block:: bash

    sudo smbpasswd -a glab

The current password is available on the `wiki <http://www/wiki/Real-time_fMRI>`_ (only accessible from within the Gallant lab network).

During real-time scans, the ``realtimefmri collect`` command monitors ``/home/glab/.local/share/realtimefmri/scanner`` for incoming DICOM files. The directory needs to be shared with the **scanner console**. Configure the samba share by appending this section to the ``/etc/samba/smb.conf``:

.. code-block:: bash

    [realtimefmri]
    path = /home/glab/.local/share/realtimefmri/scanner
    valid users = glab
    read only = no


You will need to restart the samba service for the changes to take effect.

.. code-block:: bash

    sudo service smbd restart


Using Samba share
-----------------

Each time you use the real-time system, you will need to mount the previously-configured shared folder from the **scanner console**. This involves two steps.

1. Network mounting the shared folder using Windows
2. Configuring the scanner to write out DICOM files to that folder in real-time

Much of what we learned about our real-time setup comes from another real-time project `FIRMM <http://firmm.readthedocs.io/en/latest/>`_. We copy some of their documentation here.


Network mounting the shared folder using Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use the built in Windows tools to mount the shared folder over the network

1. Press Ctrl+Esc to open the Windows start menu, click "Computer" or "Windows Explorer"
2. Click "Tools > Map network drive", and enter ``Y:`` and ``\\[REAL-TIME COMPUTER IP]\??``
3. Make sure "use different credentials" is checked, click "OK"
4. Enter ``[REAL-TIME COMPUTER IP]\glab`` for the user name and enter the samba password and click "OK"

This process links scanner console ``Y://`` to real-time computer ``/home/glab/.local/share/realtimefmri/scanner``. These are now effectively the same folder.
You can test this connection by writing a file to the directory on one computer and making sure it appears in the corresponding directory on the other computer.


Configuring the scanner to write out DICOM files to that folder in real-time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Siemens scanners come with a useful if not-that-well-documented tool called ``ideacmdtool``. Configure ``ideacmdtool`` on the scanner console using the `instructions <http://firmm.readthedocs.io/en/latest/README_ideacmdtool/>`_ in the FIRMM documentation. First, you will need to activate the "advanced user" mode on the scanner console. The username and password are available on the `wiki <http://www/wiki/Real-time_fMRI>`_.

1. Press Ctrl-Esc to open the Windows start menu.
2. Click "Command Prompt" or "Run" and enter ``cmd`` to start the command prompt.
3. Enter ``ideacmdtool``.
4. Enter ``4`` (Online export defaults) and set the following values:

.. code-block:: bash

    target port = -1
    target path = y:
    SendBuffered OFF
    SendIMA OFF

5. Enter ``q`` (back to main ``ideacmdtool`` menu)
6. Enter ``q`` (exit ``ideacmdtool``)
