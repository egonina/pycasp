#!/usr/bin/env python

# Based on codepy's setup.py (see http://mathema.tician.de/software/codepy)

import distribute_setup
import pycasp 
distribute_setup.use_setuptools()

from setuptools import setup
import glob

setup(name="pycasp",
      version=pycasp.__version__,
      description="This is a SEJITS (selective embedded just-in-time) framework for audio content analysis, built on the ASP framework.",
      long_description="""
      See http://www.armandofox.com/geek/home/sejits/ for more about SEJITS, including links to
      publications. See http://github.com/egonina/pycasp/wiki for more about PyCASP.
      """,
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        ],

      author=u"Katya Gonina, Henry Cook, Shoaib Kamil",
      url="http://github.com/egonina/pycasp/wiki/",
      author_email="egonina@cs.berkeley.edu",
      license = "BSD",

      packages=["pycasp", "gmm_specializer", "svm_specializer"],
      package_dir={'gmm_specializer':'specializers/gmm/gmm_specializer', 'svm_specializer':'specializers/svm/svm_specializer'},
      install_requires=[
        "asp",
        "scikit_learn"
          ],
     )

