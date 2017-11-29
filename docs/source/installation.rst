Installation
============

Checkout the code from github and install using ``pip``.

.. code-block:: bash
    
    git clone https://github.com/gallantlab/realtimefmri.git
    pip install realtimefmri


Dependencies
------------

The safest way to use the module is within a python virtual environment. Create a new virtual environment and install the dependencies.


``realtimefmri`` relies on ``asyncio`` to perform some asynchronous operations. This module is only available in ``python3``.

.. code-block:: bash
    
    virtualenv <environment_name> --python=python3
    source <environment_name>/bin/activate
    
    cd realtimefmri
    pip install -r requirements.txt

    # do real-time things

    # to exit the virtual environment
    deactive 

Samba file-sharing
^^^^^^^^^^^^^^^^^^
To receive files in real-time from the scanner, you need to set up SMB/CIFS on the real-time computer. General instructions can be found here `here <https://help.ubuntu.com/community/How%20to%20Create%20a%20Network%20Share%20Via%20Samba%20Via%20CLI%20%28Command-line%20interface/Linux%20Terminal%29%20-%20Uncomplicated%2C%20Simple%20and%20Brief%20Way%21>`_, and the we provide the specific instructions below. First, install ``samba`` package:

.. code-block:: bash
    
    sudo apt-get update
    sudo apt-get install samba


Samba manages its own passwords (i.e., they are not the same as your Linux passwords), so you need to create a new samba password for your user with:

.. code-block:: bash

    sudo smbpasswd -a glab

Specifics for how to set-up file sharing for ``realtimefmri`` can be found :ref:`here <network>`


Uninstall
---------

Uninstall the package using ``pip``.

.. code-block:: bash
    
    pip uninstall realtimefmri

