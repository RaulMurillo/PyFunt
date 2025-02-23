#!/usr/bin/env python
# coding: utf-8

import os
import sys
import subprocess

'''
Original Source: https://github.com/scipy/scipy/blob/master/setup.py
'''

if sys.version_info[:2] < (2, 6) or (3, 0) <= sys.version_info[0:2] < (3, 2):
    raise RuntimeError("Python version 2.6, 2.7 (TODO: >= 3.2) required.")

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

with open('./requirements.txt') as f:
    required = f.read().splitlines()

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# This is a bit hackish: we are setting a global variable so that the main
# pyfunt __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__PUFUNT_SETUP__ = True


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of pyfunt.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pyfunt/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load pyfunt/__init__.py
        import imp
        version = imp.load_source('pyfunt.version', 'pyfunt/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='pyfunt/version.py'):
    cnt = """\
        # THIS FILE IS GENERATED FROM PYFUNT SETUP.PY\
        short_version = '%(version)s'\
        version = '%(version)s'\
        full_version = '%(full_version)s'\
        git_revision = '%(git_revision)s'\
        release = %(isrelease)s\
        if not release:\
            version = full_version\
    """
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'pyfunt'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pyfunt')
    config.add_data_files(('pyfunt', '*.txt'))

    config.get_version('pyfunt/version.py')

    return config


def setup_package():

    # Rewrite the version file every time

    write_version_py()
    cmdclass = {}

    # Figure out whether to add ``*_requires = ['numpy']``.
    # We don't want to do that unconditionally, because we risk updating
    # an installed numpy which fails too often.  Just if it's not installed, we
    # may give it a try.  See gh-3379.
    build_requires = []
    try:
        import numpy
        if (len(sys.argv) >= 2 and sys.argv[1] == 'bdist_wheel' and
                sys.platform == 'darwin'):
            # We're ony building wheels for platforms where we know there's
            # also a Numpy wheel, so do this unconditionally.  See gh-5184.
            build_requires = ['numpy>=1.7.1']
    except:
        build_requires = ['numpy>=1.7.1']

    metadata = dict(
        name="pyfunt",
        author="Daniele Ettore Ciriello",
        author_email="ciriello.daniele@gmail.com",
        version="1.1.0",
        license="MIT",
        url="https://github.com/dnlcrl/PyFunt",
        download_url="https://github.com/dnlcrl/PyFunt",
        description="Pythonic Deep Learning Framework",
        packages=['pyfunt', 'pyfunt/examples', 'pyfunt/utils', 'pyfunt/examples/residual_networks', ],
        cmdclass=cmdclass,  # {'build_ext': build_ext},
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        setup_requires=build_requires,
        install_requires=required,
        # ext_modules=extensions,
        keywords='pyfunt deep learning artificial neural network convolution',
    )

    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                               sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                                               'clean')):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scipy when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
    else:
        if (len(sys.argv) >= 2 and sys.argv[1] in ('bdist_wheel', 'bdist_egg')) or (
                'develop' in sys.argv):
            # bdist_wheel/bdist_egg needs setuptools
            import setuptools

        from numpy.distutils.core import setup

        cwd = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
            # Generate Cython sources, unless building from source release
            generate_cython()

        metadata['configuration'] = configuration

    print('setup complete')
    setup(**metadata)

if __name__ == '__main__':
    setup_package()
