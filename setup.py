import setuptools

name = 'asterion'

__version__ = None
exec(open(f'{name}/_version.py').read())

description = 'Fits the asteroseismic helium-II ionisation zone glitch ' + \
              'present in the mode frequencies of solar-like oscillators.'

packages = setuptools.find_packages(include=[name, f'{name}.*'])

author = 'Alex Lyttle'

url = 'https://github.com/alexlyttle/helium-glitch-fitter'

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

with open('requirements.txt') as file:
    install_requires = file.read().splitlines()

with open('docs/requirements.txt') as file:
    docs_require = file.read().splitlines()

setuptools.setup(
    name=name,
    version=__version__,
    description=description,
    packages=packages,
    author=author,
    url=url,
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
    },
    package_data={
        '': ['*.hdf5'],
    },
    include_package_data=True,
    python_requires='>=3.6',
    licence='MIT',
)
