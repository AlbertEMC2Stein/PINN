from setuptools import setup, find_packages

setup(
    name='PDESolver',
    version='1.0.0',
    python_requires='>=3.7',
    packages=find_packages(include=['PDESolver']),
    install_requires=[
        'tensorflow >= 2.8.0',
        'numpy >= 1.21.3',
        'matplotlib >= 3.4.3'
    ]
)
