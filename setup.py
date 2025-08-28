import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CSI",
    version="0.1",
    author="Luben M. C. Cabezas, Guilherme P. Soares",
    author_email="lucruz45.cab@gmail.com",
    description="CSI: Conformalizing Statistical Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="#",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scikit-learn",
        "tqdm",
        "normflows",
        "torch",
        "matplotlib",
        "statsmodels",
    ],
)
