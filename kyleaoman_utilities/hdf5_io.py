import h5py as h5py
import numpy as np
import multiprocessing
import os.path

class _hdf5_io():

    def __init__(self, path, fbase, ncpu=0, interval=None):
        self._path = path
        self._fbase = fbase
        self._parts = self._find_parts(self._path, self._fbase)
        self._nb_cpu = multiprocessing.cpu_count() - 1 if ncpu == 0 else ncpu
        self._interval = interval

    def _subitem_interval(self, name, parts, output, intervals):
        accumulator = []
        for part, interval in zip(parts, intervals):
            with h5py.File(part, 'r') as f:
                accumulator.append(f[name][interval[0] : interval[1]].copy())
        output.put(accumulator)
        return
            

    def __getitem__(self, name):
        items = []
        all_interval_parts = self._split_interval(name)
        all_parts = [p for p, i in zip(self._parts, all_interval_parts) if i != False]
        all_interval_parts = [i for i in all_interval_parts if i != False]
        if self._nb_cpu > 1:
            parts_split = np.array_split(all_parts, self._nb_cpu)
            interval_parts_split = np.array_split(all_interval_parts, self._nb_cpu)
            procs = []
            outputs = []
            try:
                for parts, interval_parts in zip(parts_split, interval_parts_split):
                    outputs.append(multiprocessing.Queue())
                    target = self._subitem_interval
                    args = (name, parts.tolist(), outputs[-1], interval_parts.tolist())
                    procs.append(
                        multiprocessing.Process(target=target, args=args)
                    )
                    procs[-1].start()
                for output in outputs:
                    items += output.get()
                for p in procs:
                    p.join()
            except IOError:
                self._nb_cpu = 1 #fallback to serial mode
                return self[name]
        else:
            for part, interval_part in zip(all_parts, all_interval_parts):
                with h5py.File(part, 'r') as f:
                    startslice, endslice = interval_part
                    items.append(f[name][startslice : endslice].copy())
        if len(items) == 0:
            raise KeyError("Unable to open object (Object '" + name + \
                           "' doesn't exist in file with path '" + self._path + \
                           "' and basename '" + self._fbase + "')")
        else:
            return np.concatenate(items)

    def _split_interval(self, name):
        slices = []
        start = 0
        for part in self._parts:
            with h5py.File(part, 'r') as f:
                try:
                    end = start + f[name].shape[0]
                except KeyError:
                    slices.append(False)
                    continue
                if self._interval != None:
                    if self._interval[0] <= start:
                        startslice = 0
                    elif self._interval[0] <= end:
                        startslice = self._interval[0] - start
                    else:
                        slices.append(False)
                        start = end
                        continue
                    if self._interval[1] >= end:
                        endslice = end - start
                    elif self._interval[1] >= start:
                        endslice = self._interval[1] - start
                    else:
                        slices.append(False)
                        start = end
                        continue
                else:
                    startslice = 0
                    endslice = end - start
                slices.append((startslice, endslice))
                start = end
        return slices
        
    def _find_parts(self, path, fbase):
        if os.path.exists(path + '/' + fbase + '.hdf5'):
            return [path + '/' + fbase + '.hdf5']
        elif os.path.exists(path + '/' + fbase + '.0.hdf5'):
            fcount = 0
            retval = []
            while os.path.exists(path + '/' + fbase + '.' + str(fcount) + '.hdf5'):
                retval.append(path + '/' + fbase + '.' + str(fcount) + '.hdf5')
                fcount += 1
            return retval
        else:
            raise IOError("Unable to open file (File with path '" + path + \
                          "' and basename '" + fbase + "' doesn't exist)")
            
    def get_parts(self):
        return self._parts

def hdf5_get(path, fbase, hpath, attr=None, ncpu=0, interval=None):
    '''
    path: directory containing hdf5 file
    fbase: filename (omit '.X.hdf5' portion)
    hpath: 'internal' path of data table to gather, e.g. '/PartType1/ParticleIDs'
    attr: name of attribute to fetch (optional)
    ncpu: read in parallel with the given cpu count (0 -> all cpus, the default)
    interval: read a subset of a dataset in the given interval (2-tuple) of indices
    '''
    if not attr:
        hdf5_file = _hdf5_io(path, fbase, ncpu=ncpu, interval=interval)
        retval = hdf5_file[hpath]
        return retval
    else:
        for fname in _hdf5_io(path, fbase, ncpu=ncpu).get_parts():
            with h5py.File(fname, 'r') as f:
                try:
                    return f[hpath].attrs[attr]
                except KeyError:
                    continue
        raise KeyError("Unable to open attribute (One of object '" + hpath + \
                       "' or attribute '" + attr + "' doesn't exist in file with path '" + path + \
                       "' and basename '" + fbase + "')")

"""
#TESTS ON CAVI

import time

path = '/sraid14/azadehf/LG/data_fix/V1_LR_fix/particledata_127_z000p000'
fbase = 'eagle_subfind_particles_127_z000p000'
hpath = '/PartType1/Coordinates'

interval = (20000, 100000)

t0 = time.clock()
d1 = hdf5_get(path, fbase, hpath, ncpu=1)[interval[0] : interval[1]]
t1 = time.clock()
d2 = hdf5_get(path, fbase, hpath, ncpu=2)[interval[0] : interval[1]]
t2 = time.clock()
d3 = hdf5_get(path, fbase, hpath, ncpu=1, interval=interval)
t3 = time.clock()
d4 = hdf5_get(path, fbase, hpath, ncpu=2, interval=interval)
t4 = time.clock()

print(interval[1] - interval[0] == d1.shape[0])
print((d1 == d2).all(), (d1 == d3).all(), (d1 == d4).all())
print(t1-t0 > t2-t1, t2 - t1 > t3-t2, t3-t2 > t4-t3)


"""
