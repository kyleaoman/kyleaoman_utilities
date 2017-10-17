from cPickle import load
from cPickle import dump
try:
    import tables
    have_pytables = True
except ImportError:
    have_pytables = False
import numpy as np

def savevars(varlist, fname, mode='pickle'):
    if mode == 'pickle':
        if not fname[-4:] == '.pkl':
            fname += '.pkl'
        with open(fname, 'wb') as f:
            for i in varlist:
                dump(i,f,2)
    elif mode == 'hdf5':
        if not fname[-5:] == '.hdf5':
            fname += '.hdf5'
        try:
            for i in range(len(varlist)):
                if type(varlist[i]) != type(np.array([0])):
                    raise TypeError
        except TypeError:
            print 'savevars warning: use hdf5 mode for numpy arrays only! trying to pickle instead...'
            savevars(varlist, fname[:-5], mode='pickle')
            return
        if have_pytables == False:
            print 'savevars warning: PyTables not installed, pickling instead...'
            savevars(varlist, fname[:-5], mode='pickle')
            return
        with tables.openFile(fname, mode='w', title='None') as f:
            for i in range(len(varlist)):
                f.createArray(f.root, 'arr_'+'{:03d}'.format(i), varlist[i])
        return
    else:
        print 'savevars error: unknown mode! no file created.'
        return

def loadvars(fname, mode='pickle'):
    if mode == 'pickle':
        if not fname[-4:] == '.pkl':
            fname += '.pkl'
        with open(fname, 'rb') as f:
            ret = []
            try:
                while True:
                    ret.append(load(f))
            except EOFError:
                pass
        return ret
    elif mode == 'hdf5':
        if have_pytables == False:
            print 'loadvars error: PyTables not installed, cannot read .hdf5 file: nothing read.'
            return
        ret = []
        if not fname[-5:] == '.hdf5':
            fname += '.hdf5'
        with tables.openFile(fname, mode='r') as f:
            i = f.root.__iter__()
            while True:
                try:
                    ret.append(np.asarray(i.next()))
                except StopIteration:
                    break
        return ret
    else:
        print 'loadvars error: unknown mode! nothing read.'
        return
