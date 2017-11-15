import h5py as h5py
import numpy as np
import multiprocessing
import os.path

class _hdf5_io():

    def __init__(self, path, fbase, ncpu=0):
        self.__path = path
        self.__fbase = fbase
        self.__parts = self.__find_parts(self.__path, self.__fbase)
        self.__nb_cpu = multiprocessing.cpu_count() - 1 if ncpu == 0 else ncpu

    def __subitem(self, name, parts, output):
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
        if self.__nb_cpu > 1:
            try:
                parts_split = np.array_split(self.__parts, self.__nb_cpu)
                procs = []
                outputs = []
                for parts in parts_split:
                    outputs.append(multiprocessing.Queue())
                    procs.append(
                        multiprocessing.Process(
                            target=self.__subitem, 
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
                self.__nb_cpu = 1 #fallback to serial mode
                return self[name]
        else:
            items = []
            for part in self.__parts:
                with h5py.File(part, 'r') as f:
                    try:
                        items.append(f[name].value.copy())
                    except KeyError:
                        pass
        if len(items) == 0:
            raise KeyError("Unable to open object (Object '" + name + \
                           "' doesn't exist in file with path '" + self.__path + \
                           "' and basename '" + self.__fbase + "')")
        else:
            return np.concatenate(items)

    def __find_parts(self, path, fbase):
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
        return self.__parts

def hdf5_get(path, fbase, hpath, attr=None, ncpu=0):
    '''
    path: directory containing hdf5 file
    fbase: filename (omit '.X.hdf5' portion)
    hpath: 'internal' path of data table to gather, e.g. '/PartType1/ParticleIDs'
    attr: name of attribute to fetch (optional)
    '''
    print(path, fbase, hpath, attr, ncpu)
    if not attr:
        hdf5_file = _hdf5_io(path, fbase, ncpu=ncpu)
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
