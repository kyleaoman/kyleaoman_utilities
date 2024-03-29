from setuptools import setup

setup(
    name='kyleaoman_utilities',
    version='0.1',
    description='Assorted personal functions and tools.',
    url='',
    author='Kyle Oman',
    author_email='kyle.a.oman@durham.ac.uk',
    license='',
    packages=['kyleaoman_utilities'],
    install_requires=['numpy', 'astropy', 'matplotlib', 'h5py'],
    include_package_data=True,
    zip_safe=False
)
