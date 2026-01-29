import os
import platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check if we are on Windows
is_windows = platform.system() == "Windows"

# Common NVCC flags
nvcc_args = [
    "-O3",
    "-std=c++17",
    "-allow-unsupported-compiler",
]

# Common Host flags
cxx_args = ["-std=c++17"]

if is_windows:
    cxx_args += [
        "/permissive-",
        "/Zc:__cplusplus",
        "/D_WIN64",
        "/NOMINMAX",
    ]
    # Pass host compiler flags to NVCC correctly on Windows
    for flag in ["/permissive-", "/Zc:__cplusplus", "/D_WIN64", "/NOMINMAX"]:
        nvcc_args.append("-Xcompiler")
        nvcc_args.append(flag)
else:
    # Linux specific flags (if needed)
    cxx_args += ["-Wno-sign-compare"] 

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/grayscale.cu",
                "kernels/box_blur.cu",
                "kernels/sobel.cu",
                "kernels/dilation.cu",
                "kernels/erosion.cu",
                "kernels/module.cpp",
                "kernels/connectedComponents.cu",
            ],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)