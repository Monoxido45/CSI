import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CP2LFI",
    version="0.1",
    author="Luben Miguel Cruz Cabezas",
    author_email="lucruz45.cab@gmail.com",
    description="CP2LFI: Conformal Prediction applied to Likelihood Free Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="#",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scikit-learn", "tqdm", "normflows", "torch"],
)
