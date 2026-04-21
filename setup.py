from setuptools import find_packages, setup


setup(
    name="so101-hackathon",
    version="0.1.0",
    description="Student-friendly SO101 teleoperation baselines for Isaac Lab.",
    packages=find_packages(include=["so101_hackathon", "so101_hackathon.*"]),
    include_package_data=True,
    package_data={
        "so101_hackathon": [
            "assets/robots/*.usd",
            "assets/scenes/kitchen_with_orange/scene.usd",
            "assets/scenes/kitchen_with_orange/assets/textures/*.png",
            "assets/scenes/kitchen_with_orange/objects/*/*.usd",
            "configs/*.yaml",
        ]
    },
    install_requires=[
        "PyYAML>=6.0",
        "rsl-rl-lib==5.0.1",
        "tqdm>=4.66",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "so101-hackathon-train=so101_hackathon.rl_training.train_ppo:main",
            "so101-hackathon-evaluate=so101_hackathon.evaluation.evaluate:main",
            "so101-hackathon-play=so101_hackathon.evaluation.evaluate:play_main",
            "so101-hackathon-list-controllers=so101_hackathon.registry:cli_main",
        ],
    },
)
