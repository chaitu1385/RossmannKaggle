from setuptools import setup, find_packages

setup(
    name="forecasting-product",
    version="0.1.0",
    description="Modular time series forecasting product for store sales",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "pyyaml>=5.4.0",
        "joblib>=1.1.0",
    ],
)
