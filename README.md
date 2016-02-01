# realtimefmri

Real-time collection and preprocessing of functional magnetic resonance imaging data along with stimulus presentation.

# Usage

This software consists of three main bodies of code:

1. ## Data collection

The fMRI system consists of several computers operating on a local network: the *scanner computer* controlling the scanner itself, the *reconstruction computer*, and the *scanner console*. Of these, the operator only interacts directly with the *scanner console*. To engage in real-time experiments, we need to first connect the *real-time computer* to this network. This is done by connecting it via ethernet to the router switch on top of the scanner computer. This allows the *real-time computer* to access the image files as they appear in a shared directory on the *scanner console*. Once the *real-time computer* is plugged in, the following code will mount the shared folder as a drive on the *real-time computer*:

`$ mount -t cifs //192.168.2.1/IMAGES /mnt/scanner/ -o user=meduser`

(When prompted, enter the password: `meduser1`)

`collect.py` is responsible for monitoring the shared directory and can be run with:

`$ python collect.py`

This code sits on a folder and continuously checks for new images coming off of the scanner. When an image appears, it's data is sent over a zmq messaging socket to the preprocessing script.

2. ## Preprocessing

The data arriving from the data collection stage is formatted as a raw binary string of `uint16` values in the shape of a Siemens "mosaic" image. A few stages of preprocessing are needed to make it usable.

The overall preprocessing pipeline is specified in a `YAML` configuration file. The configuration file should contain at least one dictionary with the key `pipeline` and the value of a **list of steps**. An individual step has the format shown in this example...

  - name: raw_to_nifti
    instance: !!python/object:realtimefmri.core.preprocessing.RawToNifti {}
    kwargs:
      subject: *SUBJECT
    input:
      - raw_image_binary
    output:
      - image_nifti

- *Required keys:*
  - `name` (string): a descriptive name for this step
  - `instance` (python object): `YAML` tag specifying a python object (that subclasses `PreprocessingStep`), which executes the step
  - `input` (string): the name of the 

 A few `YAML` tricks are employed here, the most important of which is the ability to specify a `python` object in the configuration file using the `YAML` tag `!!python/object`. 

Preprocessing steps are subclasses of the parent class, `PreprocessingStep`. The only requirement is that they implement a `.run()` method.