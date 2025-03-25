from setuptools import setup, find_packages

setup(
    name='PDESolver',
    version='1.2.0',
    python_requires='>=3.11',
    packages=find_packages(include=['PDESolver']),
    install_requires=[
        'tensorflow_cpu >= 2.19.0',
        'numpy >= 2.1.0',
        'matplotlib >= 3.10.0',
        'tqdm >= 4.67.0',
    ]
)
