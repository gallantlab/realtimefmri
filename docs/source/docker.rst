Using ``docker-compose``
============

Installation
------------

Assuming you have docker installed, install ``docker-compose`` following the instructions `here https://docs.docker.com/compose/install/`.


Configuration
-------------
Modify the ``.env`` file with the appropriate paths and device number. In particular, set the following variables to point to

- ``PYCORTEX_STORE``: pycortex store with the subjects' surfaces and transforms
- ``PIPELINE_PATH``: path where pipelines are stored
- ``TEST_DATASET_PATH``: path where the test dataset is stored
- ``EVENT_DEVICE``: keyboard input device

Running ``docker-compose``
--------------------------

From the path containing the ``docker-compose.yml`` file, run the following command:

.. code-block:: bash
   
   docker-compose up -d

It will pull the ``redis`` docker image, as well as build the ``realtimefmri`` image, and finally start both containers. To stop them, run

.. code-block:: bash
   
   docker-compose down
