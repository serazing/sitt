from setuptools import setup

setup(name='sitt',
      version='0.1',
      description='Utilies used during my sitt at LEGOS',
      author='Guillaume SERAZIN',
      author_email='guillaume.serazin@legos.obs-mip.fr',
      packages=['sitt'],
      package_data={'sitt': ['data/*']},
      zip_safe=False)
