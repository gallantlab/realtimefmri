Installation
============

Checkout the code from github and install using ``pip``.

.. code-block:: bash
    
    git clone https://github.com/gallantlab/realtimefmri.git
    pip install realtimefmri


Using ``docker-compose``
------------------------

We recommend using `docker <https://docs.docker.com/install/>`_ and `docker-compose <https://docs.docker.com/compose/install/>`_ to run ``realtimefmri``.

Configuration
~~~~~~~~~~~~~
Modify the ``.env`` file with the appropriate paths and device number. In particular, set the following variables to point to

- ``PYCORTEX_STORE``: path to pycortex store containing subject surfaces and transforms
- ``PIPELINE_PATH``: path where pipelines are stored
- ``TEST_DATASET_PATH``: path where the test dataset is stored
- ``EVENT_DEVICE``: keyboard input device

Running ``docker-compose``
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the path containing the ``docker-compose.yml`` file, run the following command:

.. code-block:: bash
   
   docker-compose up -d

It will pull the ``redis`` docker image, build the ``realtimefmri`` image, and start both containers. To stop them, run

.. code-block:: bash
   
   docker-compose down
