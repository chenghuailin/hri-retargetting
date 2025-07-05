from setuptools import setup, find_packages

setup(
    name="HRI_retarget",
    version="0.1.0",
    packages=find_packages(),  # 自动发现所有包
    # 或者显式指定包:
    # packages=["HRI_retarget", "HRI_retarget.collision", "HRI_retarget.deploy"],
    install_requires=[
        # 你的依赖项
    ],
)