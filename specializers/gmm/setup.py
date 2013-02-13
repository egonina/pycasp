#!/usr/bin/env python

# Based on codepy's setup.py (see http://mathema.tician.de/software/codepy)

import distribute_setup
import gmm_specializer
distribute_setup.use_setuptools()

from setuptools import setup
import glob

setup(name="gmm_specializer",
      version=gmm_specializer.__version__,
      description="This is a SEJITS (selective embedded just-in-time specializer) for Gaussian Mixture Models, built on the ASP framework.",
      long_description="""
      See http://www.armandofox.com/geek/home/sejits/ for more about SEJITS, including links to
      publications. See http://github.com/hcook/gmm/wiki for more about the GMM specializer.
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

      author=u"Henry Cook, Katya Gonina, Shoaib Kamil",
      url="http://github.com/hcook/gmm/wiki/",
      author_email="egonina@cs.berkeley.edu",
      license = "BSD",

      packages=["gmm_specializer"],
      install_requires=[
        "asp",
        "scikit_learn"
          ],
     )

