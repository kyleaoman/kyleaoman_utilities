from setuptools import setup

setup(
    name='utilities',
    version='0.1',
    description='Assorted personal functions and tools.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='',
    packages=['utilities'],
    install_requires=['numpy', 'astropy', 'matplotlib'],
    include_package_data=True,
    zip_safe=False
)
