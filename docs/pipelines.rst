Configuring a real-time pipeline
=====================================

The flexibility of this package comes from your ability to configure custom preprocessing and stimulation pipelines. Pipelines are specified using :mod:`yaml` files stored in the ``pipelines`` directory.

The configuration file should contain at least one dictionary with the key `pipeline` and the value of a **list** of steps. An individual step has the following format:

* *Required keys:*
  * ``name`` (string): a descriptive name for this step
  * ``instance`` (python object): ``yaml`` tag specifying a python object (one that subclasses `PreprocessingStep`), which executes the step
  * ``input`` (list of strings): string keys to the ``data_dict`` specifying which of the entries should be passed as inputs to the ``.run()`` method for this step
* *Optional keys:*
  * ``kwargs`` (dictionary): values provided upon initialization of the step
  * ``output`` (list of strings): keys that will be paired with returned values and added to the ``data_dict`` when this step is finished
  * ``send`` (list of strings): keys indicating values from ``data_dict`` that will be sent to the stimulation code when this step is finished

Here is an example of a simple preprocessing step, the first step in fact, which converts the ``raw_image_binary`` to a more usable Nifti volume.

.. code-block:: yaml

  - name: raw_to_nifti
    instance: !!python/object:realtimefmri.preprocessing.RawToNifti {}
    kwargs:
      subject: *SUBJECT
    input:
      - raw_image_binary
    output:
      - image_nifti

As you can see, no ``send`` value is configured, meaning none of this data will be sent over to the stimulation code. However, if you wanted to (i.e. you build some stimulus that used the raw image volume) you would add the key:

.. code-block:: yaml
  send:
    - image_nifti
