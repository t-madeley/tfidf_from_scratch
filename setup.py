from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    install_requires = [req for req in fh.read().splitlines() if "==" in req]

setup(
    name="hiscox_tfidf",
    version="0.0.1",
    description="Hiscox TF-IDF From Scratch Take Home Task",
    author="t-madeley",
    packages=find_packages(),
    install_requires=install_requires,
    tests_require=["pytest", "pytest-cov"],
    python_requires="==3.11.*",
    package_data={"hiscox_tfidf": ["conf/*.*"]},
    include_package_data=True,
    entry_points={"console_scripts": ["hiscox_tfidf=hiscox_tfidf.__main__:cli"]}

)
