from setuptools import setup, find_packages

with open("README.md", mode="r") as f:
    long_description = f.read()

with open("requirements.txt", mode="r") as f:
    requirements = f.readlines()


setup(
    name="slomet-atomic-2020",
    version="0.1.0",
    description="The SLOmet-atomic 2020 experiment source code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[req for req in requirements if req[:2] != "# "],
    setup_requires=["black"]
)