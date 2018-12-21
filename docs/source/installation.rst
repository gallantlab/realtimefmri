Installation
============

Install `docker <https://docs.docker.com/install/>`_ and `docker compose <https://docs.docker.com/compose/>`_.

1. Check out the code from github.

.. code-block:: bash
    
    git clone https://github.com/gallantlab/realtimefmri.git
    cd realtimefmri


2. Modify the ``.env`` file with the appropriate paths and device number. In particular, set the following variables to point to:

- ``PYCORTEX_STORE``: directory containing pycortex data, e.g., subject surfaces and transforms
- ``EVENT_DEVICE``: keyboard input device
- ``PIPELINE_PATH``: directory containing pipelines configuration files
- ``TEST_DATASET_PATH``: directory containing test dataset
- ``DATASTORE_PATH``: directory containing auxiliary data such as pre-trained decoders
- ``STATIC_PATH``: directory containing static assets for the web interface

3. Build the docker images.

.. code-block:: bash
   
   make docker.build


After you have built the docker containers, you can run ``realtimefmri`` by opening a terminal, navigating the root directory of this repository, and running:


.. code-block:: bash
   
   make docker.up


To stop the program, run:

.. code-block:: bash
   
   make docker.down
