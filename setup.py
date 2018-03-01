from setuptools import setup, find_packages

setup(
    name='predict-client',
    version='1.7.1-rc2',
    description='Client used to send grcp requests to a tfserving model',
    url='https://github.com/epigramai/tfserving_predict_client',
    author='Stian Lind Petlund',
    author_email='stian@epigram.ai',
    packages=find_packages('.'),
    install_requires=['grpcio>=0.15.0', 'numpy==1.13.1']
)
