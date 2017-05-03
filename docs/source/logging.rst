Logging
=======

The python ``logging`` module  is used to generate log entries that provide a record of what occurred during the real-time experiment. The log is stored in ``recordings/<recording_id>/recording.log``. Here is an example log file:

::

    2017-03-30 14:12:15,733 root                 INFO     starting synchronizer
    2017-03-30 14:12:15,734 root                 INFO     starting scanner
    2017-03-30 14:12:15,734 root                 INFO     starting collector
    2017-03-30 14:12:15,734 collector            INFO     data collector initialized
    2017-03-30 14:12:15,736 root                 INFO     starting tasks
    2017-03-30 14:12:23,605 scanner              INFO     TR 1490908343.6054802
    2017-03-30 14:12:26,388 collector            INFO     volume 80d717bc-c6d1-4ff5-b6df-1ecd70f48f8c.PixelData
    2017-03-30 14:12:34,533 scanner              INFO     TR 1490908354.5333643
    2017-03-30 14:12:37,215 collector            INFO     volume f07c8b9c-466d-4d53-994a-c3417f75a385.PixelData
    2017-03-30 14:12:41,101 scanner              INFO     TR 1490908361.1013849
    2017-03-30 14:12:42,328 collector            INFO     volume e0a979a3-2003-4c95-92d5-d2fbe4ba38e7.PixelData
    2017-03-30 14:12:44,845 scanner              INFO     TR 1490908364.8452926
    2017-03-30 14:12:46,339 collector            INFO     volume 8b929de3-ecdc-401d-8c1f-343fdafca125.PixelData
    2017-03-30 14:12:49,493 scanner              INFO     TR 1490908369.4933078
    2017-03-30 14:12:50,951 collector            INFO     volume faf86960-5c46-4b05-acf9-202e56db3eb9.PixelData
    2017-03-30 14:12:53,813 scanner              INFO     TR 1490908373.81325
    2017-03-30 14:12:55,263 collector            INFO     volume 392a502f-ba92-4498-8e43-24e14480902e.PixelData
    2017-03-30 14:12:57,501 scanner              INFO     TR 1490908377.5011406
    2017-03-30 14:12:58,170 collector            INFO     volume 7fbdf386-4619-4910-985f-6d09e6240d10.PixelData
    2017-03-30 14:12:59,069 scanner              INFO     TR 1490908379.0690863
    2017-03-30 14:12:59,975 collector            INFO     volume 9335247b-a0af-4b8a-902f-a4b55802b3b5.PixelData
    2017-03-30 14:13:01,053 scanner              INFO     TR 1490908381.0530667
    2017-03-30 14:13:01,779 collector            INFO     volume a98b0f65-6a1c-4fd7-a511-5eab883da94f.PixelData
    2017-03-30 14:13:02,989 scanner              INFO     TR 1490908382.9890192
    2017-03-30 14:13:03,684 collector            INFO     volume 8ea49210-4c3f-4fe5-92ba-241dfd3f97ea.PixelData
    2017-03-30 14:13:04,925 scanner              INFO     TR 1490908384.9250386
    2017-03-30 14:13:05,689 collector            INFO     volume 5e16647e-669f-4823-9cab-6a6fd75ee68a.PixelData
    2017-03-30 14:13:07,525 scanner              INFO     TR 1490908387.5251012
    2017-03-30 14:13:08,196 collector            INFO     volume 8475b0f2-f3af-4ec5-8187-ee6082aee07f.PixelData
    2017-03-30 14:13:16,512 root                 INFO     shutting down
