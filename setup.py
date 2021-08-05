from distutils.core import setup
from setuptools import find_packages



setup(
  name = 'trucks-and-drones',
  packages=find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'A gym environment that simulates the travelling salesman or vehicle routing problem with drones or robots.',
  author = 'Maik Schürmann',
  author_email = 'maik.schuermann97@gmail.com',
  url = 'https://github.com/maik97/trucks-and-drones',
  download_url = 'https://github.com/maik97/trucks-and-drones/archive/refs/tags/v_0.0.4.tar.gz',
  keywords = ['tsp', 'vrp', 'tsp-d', 'vrp-d', 'travelling salesman problem',
              'vehicle routing problem', 'rl', 'reinforcement learning', 'gym environment'],
  install_requires=[
          'gym',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.6',
  ],
)