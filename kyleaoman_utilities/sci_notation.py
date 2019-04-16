def sci_notation(f, precision=1):
    s = ('{0:.'+'{:d}'.format(precision)+'e}').format(f)
    b, e = s.split('e')
    return r'{0:s} \times 10^{{{1:d}}}'.format(b, int(e))
