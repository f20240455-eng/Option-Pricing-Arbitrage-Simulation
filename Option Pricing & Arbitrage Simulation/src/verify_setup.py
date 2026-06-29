import sys

def verify_strike_squad_setup():
    """
    Quick check to make sure Strike Squad's environment is ready to roll.
    Run this after installing requirements.
    """
    
    print("=" * 60)
    print("STRIKE SQUAD - Environment Verification")
    print("=" * 60)
    print()
    
    issues = []
    
    # Check Python version
    print("✓ Checking Python version...")
    if sys.version_info < (3, 9):
        issues.append("Python 3.9+ required. You have: " + sys.version)
        print(f"  ⚠️  Python {sys.version_info.major}.{sys.version_info.minor} (need 3.9+)")
    else:
        print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor} - Good to go!")
    print()
    
    # Check required packages
    required_packages = [
        'pandas',
        'numpy', 
        'scipy',
        'matplotlib',
        'seaborn',
        'yfinance',
        'jupyter',
        'yaml',
        'pytest'
    ]
    
    print("✓ Checking required packages...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
            print(f"  ❌ {package} - NOT FOUND")
    print()
    
    # Summary
    if issues:
        print("=" * 60)
        print("⚠️  ISSUES FOUND:")
        print("=" * 60)
        for issue in issues:
            print(f"  • {issue}")
        print()
        print("FIX: Run this command:")
        print("  pip install -r requirements.txt")
        print()
    else:
        print("=" * 60)
        print("🎉 ALL SYSTEMS GO, STRIKE SQUAD!")
        print("=" * 60)
        print()
        print("Ready to start the HDFCBANK analysis.")
        print("Next step: Run the data fetch notebook!")
        print()
    
    return len(issues) == 0

if __name__ == "__main__":
    verify_strike_squad_setup()