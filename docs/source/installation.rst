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
    
    virtualenv <environment_name> --python=python3.5
    source <environment_name>/bin/activate
    
    cd realtimefmri
    pip install -r requirements.txt

    # do real-time things

    # to exit the virtual environment
    deactive 


Uninstall
---------

Uninstall the package using ``pip``

.. code-block:: bash
    
    pip uninstall realtimefmri

