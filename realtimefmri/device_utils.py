#!/usr/bin/env python3
import re
import asyncio
import evdev


def list_devices():
    devices = [evdev.InputDevice(dev) for dev in evdev.list_devices()]
    device_pattern = re.compile(r'/dev/input/event(\d*)')

    for device in devices:
        matches = device_pattern.match(device.fn)
        if matches:
            device.number = int(matches.groups()[0])
        else:
            device.number = None

    devices = sorted(devices, key=lambda x: x.number)

    for device in devices:
        print("{:>20}\t{}".format(device.fn, device.name))


@asyncio.coroutine
def keyboard(device_file):
    device = evdev.InputDevice(device_file)
    while True:
        events = yield from device.async_read()
        for event in events:
            event = evdev.categorize(event)
            if (isinstance(event, evdev.KeyEvent) and
               (event.keystate == event.key_down)):
                print("{}\t{}".format(device.fn, device.name))


def run():
    loop = asyncio.get_event_loop()
    devices = [evdev.InputDevice(dev) for dev in evdev.list_devices()]
    return loop.run_until_complete(asyncio.gather(*[keyboard(d.fn) for d in devices]))


if __name__ == "__main__":
    run()
