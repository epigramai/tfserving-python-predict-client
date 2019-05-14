from setuptools import setup, find_packages

setup(
    name='predict-client',
    version='1.7.2',
    description='Client used to send grcp requests to a tfserving model',
    url='https://github.com/epigramai/tfserving_predict_client',
    author='Stian Lind Petlund',
    author_email='stian@epigram.ai',
    packages=find_packages('.'),
    install_requires=['grpcio==1.20.1', 'numpy>=1.13.1']
)
