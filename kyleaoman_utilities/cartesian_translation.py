from astropy.coordinates import CartesianRepresentation, CartesianDifferential

"""
Functions which can be used to extend the CartesianRepresentation and
CartesianDifferential classes from astropy.coordinates. These should be
imported, then use:

# Extend CartesianRepresentation to allow coordinate translation
setattr(CartesianRepresentation, 'translate', translate)

# Extend CartesianDifferential to allow velocity (or other differential)
# translation
setattr(CartesianDifferential, 'translate', translate_d)
"""


def translate(cls, translation_vector):
    """
    Apply a coordinate translation.

    Parameters
    ----------
    cls : astropy.coordinates.CartesianRepresentation
        Equivalent to the 'self' argument for methods.

    translation_vector : astropy.units.Quantity, with dimensions of length
        3-vector by which to translate.

    Returns
    -------
    out : astropy.coordinates.CartesianRepresentation
        A new CartesianRepresentation instance with translation applied.
    """

    return CartesianRepresentation(
        cls.__class__.get_xyz(cls) + translation_vector.reshape(3, 1),
        differentials=cls.differentials
    )


def translate_d(cls, translation_vector):
    """
    Apply a differential translation.

    Parameters
    ----------
    cls : astropy.coordinates.CartesianDifferential
        Equivalent to the 'self' argument for methods.

    translation_vector : astropy.units.Quantity, with dimensions of velocity
                         (or other differential)
        3-vector by which to translate.

    Returns
    -------
    out : astropy.coordinates.CartesianDifferential
        A new CartesianDifferential instance with translation applied.
    """

    return CartesianDifferential(
        cls.__class__.get_d_xyz(cls) + translation_vector.reshape(3, 1)
    )
