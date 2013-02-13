#!/usr/bin/env python

# Based on codepy's setup.py (see http://mathema.tician.de/software/codepy)

import distribute_setup
import svm_specializer
distribute_setup.use_setuptools()

from setuptools import setup
import glob

setup(name="svm_specializer",
      version=svm_specializer.__version__,
      description="This is a SEJITS (selective embedded just-in-time specializer) for Support Vector Machines, built on the ASP framework.",
      long_description="""
      See http://www.armandofox.com/geek/home/sejits/ for more about SEJITS, including links to
      publications. See http://github.com/egonina/svm/wiki for more about the SVM specializer.
      """,
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        ],

      author=u"Katya Gonina",
      url="http://github.com/egonina/svm/wiki/",
      author_email="egonina@eecs.berkeley.edu",
      license = "BSD",

      packages=["svm_specializer"],
      install_requires=[
        "asp",
        "scikit_learn"
          ],
     )
