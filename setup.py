from setuptools import setup, find_packages

setup(
    name="dwl_fallguard",
    version="0.1.0",
    description="Uncertainty-aware fall detection, diagnosis, and recovery for humanoid robots",
    author="Ali El Haouari",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)