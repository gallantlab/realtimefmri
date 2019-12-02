Installation
============

Install `docker <https://docs.docker.com/install/>`_ and `docker compose <https://docs.docker.com/compose/>`_.

1. Check out the code from github.

.. code-block:: bash
    
    git clone https://github.com/gallantlab/realtimefmri.git
    cd realtimefmri


2. Modify the ``.env`` file with the appropriate paths and device number. In particular, set the following variables to point to:


- ``PYCORTEX_STORE`` (default ``./data/pycortex_store``): directory containing pycortex data, e.g., subject surfaces and transforms
- ``EVENT_DEVICE`` (default ``/dev/input/event3``): keyboard input device
- ``PIPELINE_DIR`` (default ``./realtimefmri/pipelines``): directory containing pipelines configuration files
- ``EXPERIMENT_DIR`` (default ``./realtimefmri/experiments``): 
- ``DATASTORE_DIR`` (default ``./data/datastore``): directory containing auxiliary data such as pre-trained models
- ``TEST_DATASET_DIR`` (default ``./data/test_datasets``): directory containing test dataset
- ``STATIC_PATH`` (default ``./realtimefmri/web_interface/static``): 
- ``LOG_LEVEL`` (default ``DEBUG``): directory containing static assets for the web interface


3. Build the docker images.

.. code-block:: bash
   
   make docker.build


After you have built the docker containers, you can run ``realtimefmri`` by running:


.. code-block:: bash
   
   make docker.up


To stop the program, run:

.. code-block:: bash
   
   make docker.down

And to clean up any remaining networks or volumes used by docker, run:

.. code-block:: bash

	make docker.prune
