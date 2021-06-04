import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HC4CA",
    version="0.0.7",
    author="BennyDeb",
    author_email="bennydeb@gmail.com",
    description="Hierarchical Classification for Context Awareness Pack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bennydeb/HC4CA",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/HC4CA/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=("Dataset",)),
    python_requires='>=3.6',
    install_requires=[
        'os',
        'argparse',
        'pandas',
        'sklearn',
        'os',
        'json',
        'pickle',
        'datetime',
    ],
)
