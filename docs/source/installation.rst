Installation
============

Checkout the code from github and install using ``pip``.

.. code-block:: bash
    
    git clone https://github.com/gallantlab/realtimefmri.git
    pip install realtimefmri


Dependencies
------------

The safest way to use the module is within a Python virtual environment. Create a new virtual environment and install the dependencies.


``realtimefmri`` relies on ``asyncio`` to perform some asynchronous operations. This module is only available in Python 3.

.. code-block:: bash
    
    virtualenv <environment_name> --python=python3
    source <environment_name>/bin/activate
    
    cd realtimefmri
    pip install -r requirements.txt # installs the dependencies
    pip install . # installs the realtimefmri module from setup.py

    # do real-time things

    # to exit the virtual environment
    deactive 

Network
-------

To receive files in real-time from the scanner, you need to set up Samba on the real-time computer. The real-time computer will host a shared folder, and the scanner console will write to that folder over SMB/CIFS. General instructions for setting up Samba share can be found `here <https://help.ubuntu.com/community/How%20to%20Create%20a%20Network%20Share%20Via%20Samba%20Via%20CLI%20%28Command-line%20interface/Linux%20Terminal%29%20-%20Uncomplicated%2C%20Simple%20and%20Brief%20Way%21>`_, and the we provide more specific instructions in the section :ref:`Connecting to the scanner network <network>`.


Uninstall
---------

Uninstall the package using ``pip``.

.. code-block:: bash
    
    pip uninstall realtimefmri

