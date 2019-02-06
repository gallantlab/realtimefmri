Database
========

A single redis_ database server is the hub through which the various processes communicate. Each process can write to and read from the database.

 * `Experiment`_
 * `Model`_

_`Experiment`
-------------
- ``experiment:trial``

  - Information related to a runing real-time experiment

- ``experiment:trial:current``

  - A dictionary containing the current trial ``{'start_time': <unix epoch time>, 'end_time': <unix epoch time>, 'index': <integer>}``

- ``responses:<name>:trial<trial_index>:<sample_index>``

 - A numpy array of responses corresponding to the trial ``trial_index`` and sample ``sample_index`` (sequential from the beginning of the experiment)


_`Model`
--------
 - ``model:<name>``


.. _redis: https://redis.io/documentation