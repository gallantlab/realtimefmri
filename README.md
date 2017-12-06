# realtimefmri

Real-time collection and preprocessing of functional magnetic resonance imaging data along with stimulus presentation. This software allows you to construct custom real-time data analysis pipelines without much fuss.

It consists of three parts: a **collection** system that interfaces with the Siemens MRI scanner to make brain volumes available as soon as they are acquired, a **preprocessing** system that can be configured to do common operations like motion correction and detrending, and a **stimulation** system that can present the data to the operator or subject in a variety of ways.

## 1. Data collection

## 2. Preprocessing

Images are passed through a series of preprocessing steps specified in the *preprocessing configuration file*.

### Configuring your own pipeline

Preprocessing steps are subclasses of the parent class, `PreprocessingStep`. The only requirement is that they implement a `.run()` method that performs some operation on input.

### Configuration file
The overall preprocessing pipeline is specified in a `YAML` configuration file saved in the `~/.config/realtimefmri/pipelines`. The configuration file should contain at least one dictionary that has `pipeline` and a **list** of steps. An individual step has the following format:

- *Required keys:*
  - `name` (string): a descriptive name for this step
  - `instance` (python object): `YAML` tag specifying a python object (one that subclasses `PreprocessingStep`), which executes the step
  - `input` (list of strings): string keys to the `data_dict` specifying which of the entries should be passed as inputs to the `.run()` method for this step
- *Optional keys:*
  - `kwargs` (dictionary): values provided upon initialization of the step
  - `output` (list of strings): keys that will be paired with returned values and added to the `data_dict` when this step is finished
  - `send` (list of strings): keys indicating values from `data_dict` that will be sent to the stimulation code when this step is finished

Here is an example of a simple preprocessing step, the first step in fact, which converts input Dicom image into a more usable Nifti volume.

```
- name: dicom_to_nifti
  instance: !!python/object:realtimefmri.core.preprocessing.DicomToNifti {}
  input:
    - raw_image_binary
  output:
    - image_nifti
  send:
    - image_nifti
```


The `send` field is optional. Here, it indicates that the Nifti volume should be sent along to the stimulation code.


## 3. Stimulation

Things get interesting when we make stimuli that rely on data gathered in real-time. All you need to do is build some software that manages a zmq `SUB` socket subscribed to one of the topics published by the preprocessing code. This can be implemented in any language that has has a zmq library, which is pretty much any modern language. Some generally useful stimuli are included in this library including a [pycortex](https://github.com/gallantlab/pycortex) viewer.

## Timing
The main processes (collection, preprocessing, stimulation) are able to run on separate machines, which could have different clocks. The real-time computer receives a pulse whenever a volume is acquired. This time stamp is sent alongside each volume in the code.