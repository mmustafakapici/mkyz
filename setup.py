from setuptools import setup, find_packages

setup(
    name='mkyz',
    version='0.2.1',
    packages=find_packages(),
    install_requires=[
        'pandas', 'scikit-learn', 'seaborn', 'matplotlib', 
        'mlxtend', 'numpy', 'plotly', 'statsmodels', 
        'xgboost', 'lightgbm', 'catboost', 'rich',
    ],
    description='MKYZ is a Python library for ML and data science tasks.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown', # DOĞRUSU BU
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