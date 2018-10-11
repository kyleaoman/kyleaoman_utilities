import os
import psutil

prefixes = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']


def memnow():
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss  # resident size, bytes
    ui = 0
    while rss > 1024:
        rss /= 1024
        ui += 1
    return '{:.1f} {:s}B'.format(rss, prefixes[ui])
