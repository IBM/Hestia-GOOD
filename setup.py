#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

requirements = [
    'scipy',
    'scikit-learn',
    'pandas',
    'polars',
    'numpy',
    'tqdm',
    'pyarrow'
]

test_requirements = requirements

setup(
    author="Raul Fernandez-Diaz",
    author_email='raul.fernandezdiaz@ucdconnect.ie',
    python_requires='>=3.9',
    classifiers=[
    ],
    description="Independent evaluation set construction for trustworthy ML models in biochemistry",
    entry_points={
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='hestia',
    data_files=[('hestia/utils/mmseqs_fake_prefilter.sh')],
    name='hestia-good',
    packages=find_packages(),
    long_description_content_type="text/markdown",
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/IBM/Hestia-GOOD',
    version='1.0.4',
    zip_safe=False,
)
