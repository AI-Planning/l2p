from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="l2p",
    version="0.3.3",
    packages=find_packages(exclude=["tests"]),
    description="Library to connect LLMs and planning tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marcus Tantakoun, Christian Muise",
    author_email="mtantakoun@gmail.com, christian.muise@gmail.com",
    install_requires=["retry", "pddl", "typing_extensions", "pyyaml"],
    extras_require={
        "cli": ["llm", "tiktoken", "rich"],
    },
    entry_points={
        "console_scripts": [
            "l2p=l2p.cli.main:main",
        ],
    },
    license="MIT",
    url="https://github.com/AI-Planning/l2p",
    include_package_data=True,
    package_data={
        "l2p.llm.utils": ["*"],
        "l2p.templates": ["domain_templates/*", "task_templates/*", "feedback_templates/*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
