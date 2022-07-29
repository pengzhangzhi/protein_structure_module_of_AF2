from setuptools import setup, find_packages
exec(open('structure_module/version.py').read())

setup(
  name = 'structure_module-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'structure_module = structure_module.cli:main',
      'test_structure_module = structure_module.cli:test'
    ],
  },
  version = __version__,
  license='Apache',
  description = 'pytorch implementation of structure_module of Alphafold2',
  author = 'Zhangzhi Peng',
  author_email = 'pengzhangzhics@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/pengzhangzhi',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'protein structure prediction'
  ],
  install_requires=[
    # 'einops>=0.4',
    # 'numpy',
    # 'packaging',
    # 'torch>=1.10',
    # 'tqdm',
    # "deepspeed",
    # "biopython",
    "click >= 8.1.2",
    "deepspeed >= 0.6.7",
    "dm_tree >= 0.1.7",
    "numpy >= 1.23.0",
    "scipy >=  1.8.0",
    "torch >= 1.11.0",

  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache License',
    'Programming Language :: Python :: 3.6',
  ],
)