from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="geneticprogramming",
    version="1.0.0",
    description="Genetic programming algorithm for time series clustering",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/cgp9900/Genetic_Programming",
    author="Cole Parker",
    author_email="cgp9900@gmail.com",
    keywords="genetic programming algorithm",
    license="MIT",
    packages=["genetic"],
    install_requires=[],
    include_package_data=True,
)
