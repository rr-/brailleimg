from distutils.core import Extension, setup


def main():
    setup(
        name="brailleimg",
        version="0.1",
        description="Image to braillecode converter",
        author="rr-",
        author_email="rr-@sakuya.pl",
        entry_points={
            "console_scripts": ["brailleimg = brailleimg.__main__:cli"]
        },
        install_requires=["scikit-image", "numpy", "click-pathlib", "click"],
        tests_require=["pytest"],
    )


if __name__ == "__main__":
    main()
