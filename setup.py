"""AAML Framework - Autonomous AI Machine Learning Framework with Cognitive Firewall"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aaml-framework",
    version="1.0.0",
    author="AAML Framework Contributors",
    author_email="contact@aaml.ai",
    description="Autonomous AI Machine Learning Framework with Cognitive Firewall",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaml-framework/aaml",
    project_urls={
        "Bug Tracker": "https://github.com/aaml-framework/aaml/issues",
        "Documentation": "https://docs.aaml.ai",
        "Source Code": "https://github.com/aaml-framework/aaml",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "ruff>=0.1.7",
            "mypy>=1.7.1",
            "coverage>=7.3.2",
            "locust>=2.17.0",
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ],
        "gpu": [
            "torch>=2.1.1+cu118",
            "tensorflow-gpu>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aaml-api=aaml_framework.api.enhanced_integration:main",
  # TODO: Uncomment when modules are implemented
  # "aaml-auditor=aaml_framework.cognitive_firewall.auditor_service:main",
  # "aaml-federated=aaml_framework.federated.aggregator:main",
  # "aaml-trainer=aaml_framework.training.trainer:main",

        ],
    },
    include_package_data=True,
    package_data={
        "aaml_framework": [
            "config/*.json",
            "models/*.pkl",
            "models/*.pth",
        ],
    },
)
