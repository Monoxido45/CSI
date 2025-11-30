import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CSI",
    version="0.1",
    author="Anonymous",
    author_email="",
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
        "numpy>=1.24",       # Use >= em vez de ==
        "scikit-learn>=1.3",
        "tqdm>=4.66",
        "normflows>=1.7",
        "torch>=2.0",
        "matplotlib>=3.8",
        "statsmodels>=0.14",
        "xgboost>=2.0",      # Versão 3.x ainda pode ser instável ou não existir
        "seaborn>=0.13",
        "sbibm>=0.1",
        "hypothesis"
    ],
)
