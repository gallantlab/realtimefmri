# realtimefmri

USAGE:

# Data collection:
`data_collection.DataCollector` class sits on a folder (using `data_collection.MonitorDirectory`) and continuously checks for new images coming off of the scanner. When an image appears, it's data is sent to `preprocessing.Preprocessor` over a zmq messaging socket.

# Preprocessing
The basic pipeline is as follows:
1. Convert the pixel data to a Nifti1 file
 - Since .PixelData files do not contain important metadata (scan affine), we have to provide it ahead of time.