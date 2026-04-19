from setuptools import find_packages, setup


setup(
    name="so101-hackathon",
    version="0.1.0",
    description="Student-friendly SO101 teleoperation baselines for Isaac Lab.",
    packages=find_packages(include=["so101_hackathon", "so101_hackathon.*"]),
    include_package_data=True,
    package_data={
        "so101_hackathon": [
            "sim/robots/trs_so101/LICENSE",
            "sim/robots/trs_so101/urdf/so_arm101.urdf",
            "sim/robots/trs_so101/urdf/assets/*.stl",
            "configs/*.yaml",
        ]
    },
    install_requires=[
        "PyYAML>=6.0",
        "tqdm>=4.66",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "so101-hackathon-train=so101_hackathon.training.train_ppo:main",
            "so101-hackathon-evaluate=so101_hackathon.evaluation.evaluate:main",
            "so101-hackathon-play=so101_hackathon.evaluation.evaluate:play_main",
            "so101-hackathon-list-controllers=so101_hackathon.registry:cli_main",
        ],
    },
)
