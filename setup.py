from setuptools import setup, find_packages

setup(
    name='drl_daytrading_system',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy~=1.19.2',
        'pandas~=1.3.2',
        'matplotlib~=3.4.3',
        'gym~=0.19.0',
        'tensorflow~=2.6.0'
    ]
)
