import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplespec",
    version="0.0.0",
    author="Viraj Karambelkar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/virajkaram/pydusty",
    keywords="astronomy mcmc Dusty",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.7',
    install_requires=[
        "astropy",
        "numpy",
        "emcee",
        "scipy",
        "jupyter",
        "matplotlib",
        "numpy",
        "lacosmic"
    ],
    package_data={
    }
)