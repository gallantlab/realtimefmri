.. _dashboard:

Dashboard
=========

The dashboard allows you to configure real-time visualizations of data.

Configuring the dashboard
-------------------------
To make a data element available for real-time visualization, you need to add a ``SendToDashboard`` preprocessing step to your :ref:`pipeline <pipelines>`. The ``kwargs`` must include:

- ``name``: the name of the data (used for display only)
- ``plot_type``: One of the following

  - ``bar``: a bar plot
  - ``timeseries``: a timeseries plot
  - ``array_image``: an image representation of a 2D array
  - ``static_image``: an image stored as a ``.png``, ``.jpg``, or other standard image format



Here is an example of preprocessing steps that add the z-displacement and image mosaic to the dashboard:

.. code-block:: yaml

  - name: send_motion_parameters_z
    class_name : realtimefmri.preprocess.SendToDashboard
    kwargs: { name: z_disp, plot_type: timeseries }
    input: [ z_displacement ]
  - name: send_mosaic
    class_name: realtimefmri.preprocess.SendToDashboard
    kwargs: { name: mosaic, plot_type: array_image }
    input: [ volume_mosaic ]


Using the dashboard
-------------------
After starting the :ref:`web interface <web_interface>`, visit http://localhost:8050/dashboard. Press the "↺" button to refresh the list of available data elements, then press the "+" button to add a new figure. A multi-dropdown menu will appear that allows the selection of one or several data elements to display.

.. image:: dashboard.png