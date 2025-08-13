import sys


def main():
    try:
        import pytest  # noqa: F401
    except Exception:
        print(
            "PyTest is not installed. Install it with: pip install -r requirements.txt"
        )
        sys.exit(1)

    # Run pytest
    import pytest as _pytest

    code = _pytest.main(["-q"])  # quiet output
    sys.exit(code)


if __name__ == "__main__":
    main()
