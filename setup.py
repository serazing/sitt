from setuptools import setup

setup(name='sitt',
      version='0.1',
      description='Code used for the project SWOT In The Tropics',
      author='Guillaume SERAZIN',
      author_email='guillaume.serazin@legos.obs-mip.fr',
      packages=['sitt'],
      package_data={'sitt': ['data/*']},
      zip_safe=False)
