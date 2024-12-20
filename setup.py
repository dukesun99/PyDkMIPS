from setuptools import setup, Extension, find_packages
import pybind11
import platform
import os

# Common compiler arguments from Makefile
extra_compile_args = ['-std=c++17', '-fopenmp', '-w', '-O3', '-fPIC']
extra_link_args = ['-fopenmp', '-lpthread']

ext_modules = [
    Extension(
        name="pydkmips._pydkmips_impl",
        sources=[
            os.path.join("pydkmips", "pydkmips.cpp"),
            os.path.join("DkMIPS-api", "methods_api", "pri_queue.cc"),
            os.path.join("DkMIPS-api", "methods_api", "util.cc"),
            os.path.join("DkMIPS-api", "methods_api", "bc_tree.cc"),
            os.path.join("DkMIPS-api", "methods_api", "linear.cc"),
            os.path.join("DkMIPS-api", "methods_api", "greedy.cc"),
            os.path.join("DkMIPS-api", "methods_api", "dual_greedy.cc"),
            os.path.join("DkMIPS-api", "methods_api", "bc_greedy.cc"),
            os.path.join("DkMIPS-api", "methods_api", "bc_dual_greedy.cc"),
        ],
        include_dirs=[
            pybind11.get_include(),
            os.path.join("DkMIPS-api", "methods_api")
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(
    name="pydkmips",
    version="0.1.0",
    author="Sun Yiqun",
    author_email="dukesun99@icloud.com",
    description="A Python package for DkMIPS algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        'pydkmips': ['*.so', '*.pyd'],  # Include compiled extensions
    },
    include_package_data=True,
    ext_modules=ext_modules,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 