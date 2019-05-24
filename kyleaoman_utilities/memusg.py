import os
import psutil
import resource

prefixes = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']


def memnow():
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss  # resident size, bytes
    ui = 0
    while rss > 1024:
        rss /= 1024
        ui += 1
    return '{:.12f} {:s}B'.format(rss, prefixes[ui])


def memmax():
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ui = 1
    while maxrss > 1024:
        maxrss /= 1024
        ui += 1
    return '{:.1f} {:s}B'.format(maxrss, prefixes[ui])
