from abc import ABCMeta, abstractmethod

class _CosmoParams(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, Ol=None, Om=None, Ob=None, h=None, sigma8=None, ns=None):
        self._params = {
            'Ol': Ol,
            'Om': Om,
            'Ob': Ob,
            'h': h,
            'sigma8': sigma8,
            'ns': ns
        }
        return

    def __getattr__(self, attr):
        try:
            return self._params[attr]
        except KeyError:
            raise AttributeError('Cosmological parameters are Ol, Om, Ob, h, sigma8, ns.')

    def __getitem__(self, key):
        try:
            return self._params[key]
        except KeyError:
            raise KeyError('Cosmological parameters are Ol, Om, Ob, h, sigma8, ns.')

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            return self.name == other

class WMAP7(_CosmoParams):
    
    def __init__(self):
        self.name = 'WMAP7'
        super().__init__(
            Ol=0.729, 
            Om=0.2707, 
            Ob=0.0451, 
            h=0.703, 
            sigma8=0.809, 
            ns=0.966
        )
        return

class Planck15(_CosmoParams):
    
    def __init__(self):
        self.name = 'Planck15'
        super().__init__(
            Ol=0.6911, 
            Om=0.3089, 
            Ob=0.0486, 
            h=0.6774, 
            sigma8=0.8159, 
            ns=0.9667
        )

class Planck13(_CosmoParams):
    
    def __init__(self):
        self.name = 'Planck13'
        super().__init__(
            Ol=0.693, 
            Om=0.307, 
            Ob=0.04825, 
            h=0.6777, 
            sigma8=0.8288, 
            ns=0.9611
        )

class Custom(_CosmoParams):

    def __init__(self, **kwargs):
        self.name = 'Custom'
        super().__init__()
        if set(kwargs.keys()) != set(self._params.keys()):
            raise ValueError('Custom cosmology object takes keyword arguments: Ol, Om, Ob, h, sigma8, ns.')
        self._params.update(kwargs)
        return

    
