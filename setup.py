"""Setup script for the unsloth package."""

from pathlib import Path
from setuptools import setup, find_packages


def read_readme() -> str:
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


setup(
    name="unsloth",
    version="0.1.0",
    description="Fork of unslothai/unsloth — fast LLM fine-tuning utilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Unsloth Contributors",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.38",
        "peft>=0.9",
        "bitsandbytes>=0.41",
        "accelerate>=0.27",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "ruff",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={
        "unsloth": ["models/*.md"],
    },
)
