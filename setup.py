from setuptools import setup, find_packages

setup(
    name='predict_client',
    version='1.3.1',
    description='Client used to send grcp requests to a tfserving model',
    url='https://github.com/epigramai/tfserving_predict_client',
    author='Stian Lind Petlund',
    author_email='stian@epigram.ai',
    packages=find_packages('.'),
    install_requires=['grpcio>=0.15.0']
)
