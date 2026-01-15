import os
import re
from setuptools import setup, find_packages

def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'mkyz', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='mkyz',
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        'pandas', 'scikit-learn', 'seaborn', 'matplotlib', 
        'mlxtend', 'numpy', 'plotly', 'statsmodels', 
        'xgboost', 'lightgbm', 'catboost', 'rich', 'optuna', 'scipy',
    ],
    description='MKYZ is a Python library for ML and data science tasks.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Mustafa Kapıcı',
    author_email='m.mustafakapici@gmail.com',
    url='https://github.com/mmustafakapici/mkyz',
    license_files=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)