stimulate.py
config file format

global configuration

these values are set in the __init__ method of Stimulus. if they are not configured as kwargs to individual objects, the global value will be provided. values provided as kwargs to individual objects will override global values
- subject
- record BOOL indicates if you want to record the stimulation
- recording_path STR the name of the subdirectory within $RT_DIR/recordings to which data will be recorded

if global configurations are

Stimulus objects

All stimulus objects are equipped with some class methods and attributes

- Methods
  - .stop() stops the instance

- Attributes
  - .record (default False) record output of this instance to a file. Recording functionality must be provided within subclassed objects

mount -t cifs //192.168.2.1/IMAGES /mnt/scanner/ -o user=meduser
password meduser1

# realtimefmri

USAGE:
# Data collection:
`data_collection.DataCollector` class sits on a folder (using `data_collection.MonitorDirectory`) and continuously checks for new images coming off of the scanner. When an image appears, it's data is sent to `preprocessing.Preprocessor` over a zmq messaging socket.

# Preprocessing
The basic pipeline is as follows:
1. Convert the pixel data to a Nifti1 file
 - Since .PixelData files do not contain important metadata (scan affine), we have to provide it ahead of time.

# ISSUES WHILE SCANNING
In [4]: s = pyo.Server().boot()
Expression 'parameters->channelCount <= maxChans' failed in 'src/hostapi/alsa/pa_linux_alsa.c', line: 1514
Expression 'ValidateParameters( inputParameters, hostApi, StreamDirection_In )' failed in 'src/hostapi/alsa/pa_linux_alsa.c', line: 2818
portaudio error in Pa_OpenStream: Invalid number of channels
Portaudio error: Invalid number of channels
Server not booted.

## Portaudio docs were very informative
http://portaudio.com/docs/v19-doxydocs/api_overview.html
From that I can translate the error message to mean this...
Using the ALSA (advanced linux sound architecture) Host API (the software adapter between portaudio programming interface and the platform-specific implementations), something attempted to create an audio input stream (sending info from input device to sound interface device). however the given number of channels (specified how?) did not match the expected number of channels (specified how?) for this device (which device?). perhaps assume that, since this happens when the TOSLINK SPDIF cable was connected that it was the SPDIF or iec958 device (how are these different?). also why do these not appear when we were debugging downstairs?

In [3]: pyo.pa_list_devices()
ALSA lib pcm_dsnoop.c:618:(snd_pcm_dsnoop_open) unable to open slave
ALSA lib pcm.c:2239:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
ALSA lib pcm.c:2239:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
ALSA lib pcm.c:2239:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
bt_audio_service_open: connect() failed: Connection refused (111)
bt_audio_service_open: connect() failed: Connection refused (111)
bt_audio_service_open: connect() failed: Connection refused (111)
bt_audio_service_open: connect() failed: Connection refused (111)
Cannot connect to server socket err = No such file or directory
Cannot connect to server request channel
jack server is not running or cannot be started

AUDIO devices:
0: OUT, name: HDA Intel PCH: ALC892 Analog (hw:0,0), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

1: IN, name: HDA Intel PCH: ALC892 Alt Analog (hw:0,2), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

2: OUT, name: HDA NVidia: HDMI 0 (hw:1,3), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
3: OUT, name: HDA NVidia: HDMI 1 (hw:1,7), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
4: OUT, name: HDA NVidia: HDMI 2 (hw:1,8), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
5: OUT, name: HDA NVidia: HDMI 3 (hw:1,9), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

6: OUT, name: sysdefault, host api index: 0, default sr: 48000 Hz, latency: 0.021333 s

7: OUT, name: front, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

8: OUT, name: surround40, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
9: OUT, name: surround51, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
10: OUT, name: surround71, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

11: IN, name: pulse, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s
11: OUT, name: pulse, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s

12: OUT, name: dmix, host api index: 0, default sr: 48000 Hz, latency: 0.021333 s

13: IN, name: default, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s
13: OUT, name: default, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s


NOW IN THE LAB
AUDIO devices:
0: IN, name: HDA Intel PCH: ALC892 Analog (hw:0,0), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
0: OUT, name: HDA Intel PCH: ALC892 Analog (hw:0,0), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

1: OUT, name: HDA Intel PCH: ALC892 Digital (hw:0,1), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

2: IN, name: HDA Intel PCH: ALC892 Alt Analog (hw:0,2), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

3: OUT, name: HDA NVidia: HDMI 0 (hw:1,3), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
4: OUT, name: HDA NVidia: HDMI 1 (hw:1,7), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
5: OUT, name: HDA NVidia: HDMI 2 (hw:1,8), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
6: OUT, name: HDA NVidia: HDMI 3 (hw:1,9), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

7: IN, name: sysdefault, host api index: 0, default sr: 48000 Hz, latency: 0.021333 s
7: OUT, name: sysdefault, host api index: 0, default sr: 48000 Hz, latency: 0.021333 s

8: OUT, name: front, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

9: OUT, name: surround40, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
10: OUT, name: surround51, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
11: OUT, name: surround71, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

12: OUT, name: iec958, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

13: OUT, name: spdif, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s

14: IN, name: pulse, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s
14: OUT, name: pulse, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s

15: OUT, name: dmix, host api index: 0, default sr: 48000 Hz, latency: 0.021333 s

16: IN, name: default, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s
16: OUT, name: default, host api index: 0, default sr: 44100 Hz, latency: 0.008707 s

NOW \ THEN
0: IN, name: HDA Intel PCH: ALC892 Analog (hw:0,0), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
1: OUT, name: HDA Intel PCH: ALC892 Digital (hw:0,1), host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
7: IN, name: sysdefault, host api index: 0, default sr: 48000 Hz, latency: 0.021333 s
12: OUT, name: iec958, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s
13: OUT, name: spdif, host api index: 0, default sr: 44100 Hz, latency: 0.005805 s



lspci -v
list pci (peripheral component interconnect) devices
here are the potentially audio ones

00:1b.0 Audio device: Intel Corporation Device 8ca0
        Subsystem: ASUSTeK Computer Inc. Device 863d
        Flags: bus master, fast devsel, latency 0, IRQ 45
        Memory at f7130000 (64-bit, non-prefetchable) [size=16K]
        Capabilities: <access denied>
        Kernel driver in use: snd_hda_intel

01:00.1 Audio device: NVIDIA Corporation Device 0fbb (rev a1)
        Subsystem: Micro-Star International Co., Ltd. [MSI] Device 3160
        Flags: bus master, fast devsel, latency 0, IRQ 17
        Memory at f7080000 (32-bit, non-prefetchable) [size=16K]
        Capabilities: <access denied>
        Kernel driver in use: snd_hda_intel

aplay -l
       -l, --list-devices
              List all soundcards and digital audio devices

**** List of PLAYBACK Hardware Devices ****
card 0: PCH [HDA Intel PCH], device 0: ALC892 Analog [ALC892 Analog]
  Subdevices: 0/1
  Subdevice #0: subdevice #0
card 0: PCH [HDA Intel PCH], device 1: ALC892 Digital [ALC892 Digital]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: NVidia [HDA NVidia], device 3: HDMI 0 [HDMI 0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: NVidia [HDA NVidia], device 7: HDMI 1 [HDMI 1]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: NVidia [HDA NVidia], device 8: HDMI 2 [HDMI 2]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: NVidia [HDA NVidia], device 9: HDMI 3 [HDMI 3]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

portaudio Host APIs
index: 0, id: 8, name: ALSA, num devices: 10, default in: 9, default out: 9
[(8, 'name': 'pulse'}),
 (1, 'name': 'HDA Intel PCH: ALC892 Alt Analog (hw:0,2)'}),
 (9, 'name': 'default'})]
[(0, 'name': 'HDA Intel PCH: ALC892 Digital (hw:0,1)'}),
 (2, 'name': 'HDA NVidia: HDMI 0 (hw:1,3)'}),
 (3, 'name': 'HDA NVidia: HDMI 1 (hw:1,7)'}),
 (4, 'name': 'HDA NVidia: HDMI 2 (hw:1,8)'}),
 (5, 'name': 'HDA NVidia: HDMI 3 (hw:1,9)'}),
 (6, 'name': 'iec958'}),
 (7, 'name': 'spdif'}),
 (8, 'name': 'pulse'}),
 (9, 'name': 'default'})]


index: 1, id: 7, name: OSS, num devices: 0, default in: -1, default out: -1
[None]

index: 2, id: 12, name: JACK Audio Connection Kit, num devices: 2, default in: 10, default out: 10
[(10, 'name': 'system'}),
 (11, 'name': 'pyo'})]
[(10, 'name': 'system'}),
 (11, 'name': 'pyo'})]

dsnoop
dsnoop is the equivalent of the dmix plugin, but for recording sound. The dsnoop plugin allows several applications to record from the same device simultaneously.

PCH
The Platform Controller Hub (PCH) is a family of Intel microchips, introduced circa 2008. It is the successor to the previous Intel Hub Architecture, which used a northbridge and southbridge instead, and first appeared in the Intel 5 Series. The PCH controls certain data paths and support functions used in conjunction with Intel CPUs. These include clocking (the system clock), Flexible Display Interface (FDI) and Direct Media Interface (DMI), although FDI is only used when the chipset is required to support a processor with integrated graphics. As such, I/O functions are reassigned between this new central hub and the CPU compared to the previous architecture: some northbridge functions, the memory controller and PCI-e lanes, were integrated into the CPU while the PCH took over the remaining functions in addition to the traditional roles of the southbridge. [wikipedia](https://en.wikipedia.org/wiki/Platform_Controller_Hub)

PCH is what you'd call your actual "sound card," though it isn't actually a card in the machine. PCH is Intel's name for a family of ICs (integrated circuit) what include sound output. Because there have been many generations of PCH, you often see it followed by a fragment of the IC part number, like PCH C220, which helps you decide whether a given driver is compatible with the particular PCH family variant on your motherboard.
[source](http://unix.stackexchange.com/questions/122986/how-to-understand-list-of-soundcards-meaning-of-mid-hdmi-pch)

ALSA
Advanced Linux Sound Architecture (ALSA) is a software framework and part of the Linux kernel that provides an application programming interface (API) for sound card device drivers. Some of the goals of the ALSA project at its inception were automatic configuration of sound-card hardware and graceful handling of multiple sound devices in a system. Some frameworks such as JACK use ALSA to allow performing low-latency professional-grade audio editing and mixing.

HDA
High definition audio brings consumer electronics quality sound to the PC delivering high quality sound from multiple channels. Using HDA, systems can deliver 192 kHz/32-bit quality for eight channels, supporting new audio formats.

ALC892 (Realtek)
7.1+2 Channel HD Audio Codec with Content Protection