import setuptools
import lianyhaii

with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="lianyhaii",
    version=f"{lianyhaii.__version__}",
    author="Ernnnn",
    author_email="lianyhai@163.com",
    description="A package to win data competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ernnnn/lianyhaii",
    packages=setuptools.find_packages(exclude=['build', 'dist', 'example', 'lianyhaii.egg-info']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'scikit-learn',
        'optuna',
        'plotly',
        'catboost',
        'xgboost',
        'lightgbm',
        'seaborn',
        'pyarrow',
        # 'missingno',
        # 'statsmodels',
        # 'pmdarima',
        # 'pytest',
        'gensim',
        # 'gokinjo',
        'pathos',
    ],
    include_package_data=True,
    zip_safe=False
)
#python setup.py sdist bdist_wheel
# python -m twine upload dist/*
