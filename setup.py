from setuptools import setup, find_packages

setup(
    name="benchmark_evaluator",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "sympy==1.13.3",
        "numpy==2.2.5",
        "regex==2024.11.6",
        "google-genai==1.13.0",
        "requests==2.32.2",
        "openai==1.73.0",
        "datasets==3.5.1",
        "matplotlib==3.10.0",
        "tqdm==4.67.1",
        "python-dotenv==1.1.0",
    ],
    package_data={
        "benchmark_evaluator": ["config/*.json"],  # <— include all .json under config
    },
    python_requires=">=3.12",
)