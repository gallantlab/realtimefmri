# realtimefmri

Real-time collection and preprocessing of functional magnetic resonance imaging data along with stimulus presentation.

# Usage

This software consists of three main bodies of code:

## 1. Data collection
Launch the data collection script using:

`$ python collect.py`

The fMRI system consists of several computers operating on a local network: the *scanner computer* controlling the scanner itself, the *reconstruction computer*, and the *scanner console*. Of these, the operator only interacts directly with the *scanner console*. To engage in real-time experiments, we need to first connect the *real-time computer* to this network. This is done by connecting it via ethernet to the router switch on top of the scanner computer. This allows the *real-time computer* to access the image files as they appear in a shared directory on the *scanner console*.

Visit the [wiki](http://www/wiki/Real-time_fMRI) for instructions on how to access images from the shared scanner network.

`collect.py` sits on the shared folder and continuously checks for new images coming off of the scanner. When an image appears, it's data is sent over a zmq messaging socket to the preprocessing script.

## 2. Preprocessing

Launch the preprocessing script using:

`$ python preprocess.py config_file_name`

The data arriving from the data collection stage is formatted as a raw binary string of `uint16` values in the shape of a Siemens "mosaic" image. A few stages of preprocessing are needed to make it usable. The overall structure of preprocessing is simple: the `Preprocess` object sits on the zmq port and waits for image data to arrive. When it does it is passed through a series of preprocessing steps specified in the *preprocessing configuration file*. A *dictionary of data*, conveniently known as `data_dict`, holds references to all of the data objects that need to be passed between steps or sent on to the stimulation code. Specified data will be published to a zmq `PUB` socket, to which other code can subscribe.

### `PreprocessingStep` objects
Preprocessing steps are subclasses of the parent class, `PreprocessingStep`. The only requirement is that they implement a `.run()` method that performs some operation on input.

### Configuration file
The overall preprocessing pipeline is specified in a `YAML` configuration file saved in the `~/.config/realtimefmri/pipelines`. The configuration file should contain at least one dictionary with the key `pipeline` and the value of a **list** of steps. An individual step has the following format:

- *Required keys:*
  - `name` (string): a descriptive name for this step
  - `instance` (python object): `YAML` tag specifying a python object (one that subclasses `PreprocessingStep`), which executes the step
  - `input` (list of strings): string keys to the `data_dict` specifying which of the entries should be passed as inputs to the `.run()` method for this step
- *Optional keys:*
  - `kwargs` (dictionary): values provided upon initialization of the step
  - `output` (list of strings): keys that will be paired with returned values and added to the `data_dict` when this step is finished
  - `send` (list of strings): keys indicating values from `data_dict` that will be sent to the stimulation code when this step is finished

Here is an example of a simple preprocessing step, the first step in fact, which converts the `raw_image_binary` to a more usable Nifti volume.

```
- name: raw_to_nifti`
  instance: !!python/object:realtimefmri.core.preprocessing.RawToNifti {}
  kwargs:
    subject: *SUBJECT
  input:
    - raw_image_binary
  output:
    - image_nifti
```

As you can see, no `send` value is configured, meaning none of this data will be send over to the stimulation code. However, if you wanted to (i.e. you build some stimulus that used the raw image volume) you would add the key:

```
  send:
    - image_nifti
```

## 3. Stimulation

Making stimuli that rely on data gathered in real-time; this is where things get interesting. Basically, you can make anything you can imagine. All you need to do is build some software that manages a zmq `SUB` socket subscribed to one of the topics published by the preprocessing code. This can be implemented in any code that has has a zmq library, which is pretty much any code. Some generally useful stimuli are included in this library including a [pycortex](https://github.com/gallantlab/pycortex) viewer and a dumb simple data visualization.

## Timing
The main processes (collection, preprocessing, stimulation) are able to run on separate machines, which could have different clocks. The scanner outputs a "5" keypress (via the FORP) to stimmy at the onset of each TR. After initializing the processes, each one idles until the first of these "5"s, the clock time of this message is stored and subsequent events are timestamped relative to it, providing some degree of synchronization.