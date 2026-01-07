from setuptools import setup, find_packages

setup(
    name="loosh-inference-validator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Dependencies are specified in pyproject.toml
    ],
    package_data={
        "": ["*.py"],
    },
    include_package_data=True,
    python_requires=">=3.12",
) 