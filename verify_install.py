#!/usr/bin/env python3
"""Quick verification that the package is installed correctly."""

import sys

def test_imports():
    """Test basic imports."""
    errors = []

    # Test core package
    try:
        import impact_detector
        print("✓ impact_detector package found")
    except ImportError as e:
        errors.append(f"✗ impact_detector: {e}")

    # Test modules
    modules = [
        "impact_detector.config",
        "impact_detector.data_utils",
        "impact_detector.dataset",
        "impact_detector.model",
        "impact_detector.inference",
        "impact_detector.api",
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            errors.append(f"✗ {module}: {e}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  {error}")
        return 1
    else:
        print("\n✓ All imports successful!")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())
