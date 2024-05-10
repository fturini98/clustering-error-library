from setuptools import setup, find_packages

setup(
    name='ClusteringError',
    readme = "README.md",
    version = '0.dev0',
    author='Francesco Turini',
    author_email='fturini.turini7@gmail.com',
    description='Libreria per fare il clustering di dati con errori di misura.',
    url='https://github.com/fturini98/clustering-error-library',
    license='GNU General Public License (GPL)',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'json',
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Operating System :: Microsoft :: Windows',
    ],
)