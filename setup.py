import setuptools

# with open("requirements.txt", "r") as fh:
#     requirements = fh.readlines()

setuptools.setup(
    name="ds_ada_phenotype",
    version="0.0.1",
    author="RelationRX",
    author_email="dominic@relationrx.com",
    description="Data science repository for phenotype prediction task in Ada project",
    packages=setuptools.find_packages(),
    # install_requires=[req for req in requirements if req[:2] != "# "],
)
