Logging
=======

The python ``logging`` module  is used to generate log entries that provide a record of what occurred during the real-time experiment. Collection, preprocessing, stimulation, and scanner scripts log to the same file stored in ``recordings/<recording_id>``. Here is an example log file:

::

    2017-03-13 14:26:54,909 scanner              INFO     TR
    2017-03-13 14:26:54,935 collecting           INFO     simulating 114 files from /auto/k1/robertg/code/realtimefmri/datasets/bbc_test
    2017-03-13 14:26:55,670 preprocessing        INFO     running
    2017-03-13 14:26:56,912 scanner              INFO     TR
    2017-03-13 14:26:56,942 collecting           INFO     0024202140892738141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:26:56,944 preprocessing        INFO     received image   0
    2017-03-13 14:26:57,856 stimulating          INFO     running debug
    2017-03-13 14:26:58,914 scanner              INFO     TR
    2017-03-13 14:26:59,148 collecting           INFO     0056112687594538141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:26:59,150 preprocessing        INFO     received image   1
    2017-03-13 14:27:00,071 stimulating          INFO     running debug
    2017-03-13 14:27:00,917 scanner              INFO     TR
    2017-03-13 14:27:01,354 collecting           INFO     0188166034691638141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:01,356 preprocessing        INFO     received image   2
    2017-03-13 14:27:02,271 stimulating          INFO     running debug
    2017-03-13 14:27:02,919 scanner              INFO     TR
    2017-03-13 14:27:03,560 collecting           INFO     0343261167791738141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:03,562 preprocessing        INFO     received image   3
    2017-03-13 14:27:04,451 stimulating          INFO     running debug
    2017-03-13 14:27:04,921 scanner              INFO     TR
    2017-03-13 14:27:05,767 collecting           INFO     0408161903690638141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:05,768 preprocessing        INFO     received image   4
    2017-03-13 14:27:06,614 stimulating          INFO     running debug
    2017-03-13 14:27:06,923 scanner              INFO     TR
    2017-03-13 14:27:07,972 collecting           INFO     0530291580793638141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:07,974 preprocessing        INFO     received image   5
    2017-03-13 14:27:08,874 stimulating          INFO     running debug
    2017-03-13 14:27:08,926 scanner              INFO     TR
    2017-03-13 14:27:10,178 collecting           INFO     0662205626790738141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:10,180 preprocessing        INFO     received image   6
    2017-03-13 14:27:10,927 scanner              INFO     TR
    2017-03-13 14:27:11,043 stimulating          INFO     running debug
    2017-03-13 14:27:12,384 collecting           INFO     0664286931853738141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:12,386 preprocessing        INFO     received image   7
    2017-03-13 14:27:12,929 scanner              INFO     TR
    2017-03-13 14:27:13,318 stimulating          INFO     running debug
    2017-03-13 14:27:14,590 collecting           INFO     0727187349595538141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:14,592 preprocessing        INFO     received image   8
    2017-03-13 14:27:14,931 scanner              INFO     TR
    2017-03-13 14:27:15,487 stimulating          INFO     running debug
    2017-03-13 14:27:16,796 collecting           INFO     0794224572893738141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:16,798 preprocessing        INFO     received image   9
    2017-03-13 14:27:16,933 scanner              INFO     TR
    2017-03-13 14:27:17,706 stimulating          INFO     running debug
    2017-03-13 14:27:18,935 scanner              INFO     TR
    2017-03-13 14:27:19,002 collecting           INFO     0859171526692638141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:19,004 preprocessing        INFO     received image  10
    2017-03-13 14:27:19,896 stimulating          INFO     running debug
    2017-03-13 14:27:20,938 scanner              INFO     TR
    2017-03-13 14:27:21,208 collecting           INFO     0981294426795638141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:21,210 preprocessing        INFO     received image  11
    2017-03-13 14:27:22,131 stimulating          INFO     running debug
    2017-03-13 14:27:22,940 scanner              INFO     TR
    2017-03-13 14:27:23,415 collecting           INFO     1024210260892738141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:23,416 preprocessing        INFO     received image  12
    2017-03-13 14:27:24,284 stimulating          INFO     running debug
    2017-03-13 14:27:24,942 scanner              INFO     TR
    2017-03-13 14:27:25,623 collecting           INFO     1056192138594538141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:25,625 preprocessing        INFO     received image  13
    2017-03-13 14:27:26,488 stimulating          INFO     running debug
    2017-03-13 14:27:26,944 scanner              INFO     TR
    2017-03-13 14:27:27,829 collecting           INFO     1188115344691638141216102.82353.23.2.5.7011.2.21.3.1.PixelData 720000
    2017-03-13 14:27:27,831 preprocessing        INFO     received image  14
    2017-03-13 14:27:28,727 stimulating          INFO     running debug
    2017-03-13 14:27:28,947 scanner              INFO     TR
