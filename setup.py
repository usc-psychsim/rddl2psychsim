from setuptools import setup, find_packages

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

setup(name='rddl2psychsim',
      version='0.1',
      description='Python framework for converting RDDL domains into PsychSim.',
      author='Pedro Sequeira',
      author_email='pedrodbs@gmail.com',
      url='https://github.com/usc-psychsim/rddl2psychsim',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'psychsim',
          'pyrddl @ git+https://github.com/usc-psychsim/pyrddl.git'
      ],
      zip_safe=True,
      python_requires='>=3.7',
      )
