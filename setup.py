from setuptools import setup, find_packages

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

setup(name='psychsim_rddl',
      version='1.0',
      description='Framework for converting RDDL domains into PsychSim.',
      author='Pedro Sequeira',
      author_email='pedrodbs@gmail.com',
      url='https://github.com/usc-psychsim/psychsim-rddl',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'psychsim',
          'pyrddl'
      ],
      zip_safe=True,
      python_requires='>=3.7',
      )
