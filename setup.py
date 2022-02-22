import setuptools
from ast import literal_eval

name = 'asterion'

with open(f'{name}/version.py') as file:
    # Assuming version.py follows format __version__ = '<version_string>'
    line = file.readline().strip()
    version = literal_eval(line.split(' = ')[1])

description = 'Fits the asteroseismic helium-II ionisation zone glitch ' + \
              'present in the mode frequencies of solar-like oscillators.'

packages = setuptools.find_packages(include=[name, f'{name}.*'])

author = 'Alex Lyttle'

url = 'https://github.com/alexlyttle/asterion'

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

with open('requirements.txt') as file:
    install_requires = file.read().splitlines()

with open('docs/requirements.txt') as file:
    docs_require = file.read().splitlines()

with open('tests/requirements.txt') as file:
    tests_require = file.read().splitlines()

setuptools.setup(
    name=name,
    version=version,
    description=description,
    packages=packages,
    author=author,
    url=url,
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
        'tests': tests_require,
    },
    package_data={
        '': ['*.hdf5'],
    },
    include_package_data=True,
    python_requires='>=3.8',
    licence='MIT',
)
