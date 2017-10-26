from pickle import load, dump

def savevars(varlist, fname, mode='pickle'):
    if not fname[-4:] == '.pkl':
        fname += '.pkl'
    with open(fname, 'wb') as f:
        for i in varlist:
            dump(i, f, 2)

def loadvars(fname, mode='pickle'):
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
