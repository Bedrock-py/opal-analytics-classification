"""Pip install script for bedrock.analytics.classification.*
"""

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def opalRegistration():
    from bedrock.core.opals import manage_opals
    algs = ['Knn','Svm']
    for alg in algs:
        success = manage_opals("add","analytics","opals.classification.{0}.{0}".format(alg))
        if (success == False):
            success = manage_opals("reload","analytics","opals.classification.{0}.{0}".format(alg))
            if (success == False):
                raise


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        opalRegistration()
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        opalRegistration()
        install.run(self)


setup(
    name='opals.classification',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.2',

    description='Default classification algorithms',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.gatech.edu/Bedrock/opal-analytics-classification',

    # Author details
    author='Georgia Tech Research Institute',
    author_email='trevor.goodyear@gtri.gatech.edu',

    # Choose your license
    license='GPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='analytics machinelearning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['opals.classification'],
    namespace_packages=['opals'],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'pandas',
        'bedrock'
        ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        # '': ['CONSTANTS.py','docs/*.py','docs/*.pptx'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # scripts=['bin/opal'],
    entry_points={
    },
)

0