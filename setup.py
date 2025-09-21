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
        "numpy==2.3.3",
        "scikit-learn==1.7.2",
        "tqdm==4.67.1",
        "normflows==1.7.3",
        "torch==2.8.0",
        "matplotlib==3.10.6",
        "statsmodels==0.14.5",
        "xgboost==3.0.5",
        "seaborn==0.13.2",
    ],
)
