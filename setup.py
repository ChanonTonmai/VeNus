from setuptools import setup, find_packages

setup(
    name="venus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "venus=venus.cli.main:main",
        ],
    },
    author="K. Chanon",
    author_email="tonmai12369@gmail.com",
    description="A DSL interpreter for RISC-V CGRA FPGA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-repo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 