from setuptools import setup, find_packages

setup(
    name="map_similarity_ws",
    version="0.1.0",
    author="Your Name",
    description="Wasserstein-based similarity metric for navigation environments",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "POT",
        "matplotlib"
    ],
    python_requires='>=3.8',
)