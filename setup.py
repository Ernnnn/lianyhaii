import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lianyhaii",
    version="0.0.3",
    author="Ernnnn",
    author_email="lianyhai@163.com",
    description="A package to win data competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ernnnn/lianyhaii",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'scikit-learn >= 1.0',
        'optuna',
        'plotly',
        'catboost',
        'xgboost',
        'lightgbm',
        'seaborn',
        'statsmodels',
        'pmdarima',
        'pytest',
        'gensim',
        'gokinjo',
    ],
)