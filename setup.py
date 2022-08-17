from setuptools import setup, find_packages


setup(
    name='nsnet2-denoiser',
    version="0.2.1",
    description='NSNet2 Deep Noise Suppression (DNS) package',
    url='https://github.com/NeonGeckoCom/nsnet2-denoiser',
    author='Neongecko',
    author_email='developers@neon.ai',
    license='CC-BY-4.0, MIT',
    packages=find_packages(),
    python_requires='>3.6.0',
    install_requires=[
        "onnxruntime",
        "torch",
        "numpy",
        "soundfile",
        "scipy"
    ],
    zip_safe=True,
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Speech',

        'Programming Language :: Python :: 3.6',
    ]
)