from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "datasets>=2.16.0",
    "sentencepiece>=0.1.99",
    "tqdm>=4.66.0",
    "psutil>=5.9.0",
    "accelerate>=0.26.0",
    "peft>=0.7.0",
    "bitsandbytes>=0.42.0",
    "protobuf<4.0.0",
    "huggingface_hub>=0.20.0",
    "packaging>=23.0",
    "numpy>=1.24.0",
    "triton>=2.1.0",
    "xformers>=0.0.23",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
    ],
    "vision": [
        "Pillow>=10.0.0",
        "torchvision>=0.16.0",
    ],
    "bnb": [
        "bitsandbytes>=0.42.0",
    ],
}

setup(
    name="unsloth",
    version="2024.3.0",
    author="Unsloth AI",
    author_email="danielhanchen@gmail.com",
    description="2-5x faster, 70% less memory LLM finetuning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/unslothai/unsloth",
    project_urls={
        "Bug Tracker": "https://github.com/unslothai/unsloth/issues",
        "Documentation": "https://github.com/unslothai/unsloth#readme",
        "Source": "https://github.com/unslothai/unsloth",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
    ],
    keywords= "finetuning", "lora", "ora", "transformers", learning", "deep learning", "nlp",
    ],
    license="Apache 2.0",
    include_package_data=True,
    zip_safe=False,
)
