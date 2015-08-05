
import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension(
        "vocab.vocab",
        sources=['vocab/vocab.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=['-std=c++0x'],
        language="c++"
    )
]

setup(name='vocab',
      version='0.1',
      description='Vocabulary package',
      author='Data Science',
      author_email='science@moz.com',
      packages=['vocab'],
      package_dir={'vocab': 'vocab'},
      package_data={'vocab': ['stop_words.txt']},
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
     )
