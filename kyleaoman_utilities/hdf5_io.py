import h5py as h5py
import numpy as np
import multiprocessing
import os.path
import warnings

class _hdf5_io():

    def __init__(self, path, fbase, ncpu=0, interval=False):
        self._path = path
        self._fbase = fbase
        self._parts = self._find_parts(self._path, self._fbase)
        self._nb_cpu = multiprocessing.cpu_count() - 1 if ncpu == 0 else ncpu
        self._interval = interval

    def _subitem(self, name, parts, output):
        accumulator = []
        for part in parts:
            with h5py.File(part, 'r') as f:
                try:
                    accumulator.append(f[name].value.copy())
                except KeyError:
                    pass
        output.put(accumulator)
        return

    def __getitem__(self, name):
        if self._interval != False:
            interval_parts = self._split_interval(name)
        if self._nb_cpu > 1:
            try:
                parts_split = np.array_split(self._parts, self._nb_cpu)
                procs = []
                outputs = []
                for parts in parts_split:
                    outputs.append(multiprocessing.Queue())
                    procs.append(
                        multiprocessing.Process(
                            target=self._subitem, 
                            args=(name, parts.tolist(), outputs[-1])
                        )
                    )
                    procs[-1].start()
                items = []
                for output in outputs:
                    items += output.get()
                for p in procs:
                    p.join()
            except IOError:
                self._nb_cpu = 1 #fallback to serial mode
                return self[name]
        else:
            items = []
            for pn, part in enumerate(self._parts):
                with h5py.File(part, 'r') as f:
                    if self._interval == False:
                        try:
                            items.append(f[name].value.copy())
                        except KeyError:
                            pass
                    else:
                        if interval_parts[pn] == False:
                            continue
                        else:
                            startslice, endslice = interval_parts[pn]
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
                if self._interval[0] <= start:
                    startslice = 0
                elif self._interval[0] <= end:
                    startslice = self._interval[0] - start
                else:
                    slices.append(False)
                    start = end
                    continue
                if self._interval[1] >= end:
                    endslice = end
                elif self._interval[1] >= start:
                    endslice = self._interval[1] - start
                else:
                    slices.append(False)
                    start = end
                    continue
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

def hdf5_get(path, fbase, hpath, attr=None, ncpu=0, interval=False):
    '''
    path: directory containing hdf5 file
    fbase: filename (omit '.X.hdf5' portion)
    hpath: 'internal' path of data table to gather, e.g. '/PartType1/ParticleIDs'
    attr: name of attribute to fetch (optional)
    '''
    if not attr:
        if (interval != False) and (ncpu != 1):
            warnings.warn("Using interval with hdf5_get must use ncpu=1, proceeding with serial execution.")
            ncpu = 1
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
