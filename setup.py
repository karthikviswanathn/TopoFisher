from setuptools import setup, find_packages

setup(
    name='topofisher',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'numpy',
        'gudhi',
        'powerbox',
    ],
    python_requires='>=3.8',
    author='Karthik Viswanathan',
    description='Topological Fisher Information Analysis in PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
