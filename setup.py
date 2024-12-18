import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xrd-ae",
    version="24.12.18",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="A neural network model reconstructing X-ray diffraction images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/peaknet",
    keywords = ['X-ray', 'Neural networks'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
