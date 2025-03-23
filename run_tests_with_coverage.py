import subprocess
import sys
import os
from pathlib import Path

def run_coverage():
    """Run tests with coverage and generate reports."""
    # Get the directory containing this script
    base_dir = Path(__file__).parent
    
    # Run coverage with pytest
    print("Running tests with coverage...")
    result = subprocess.run([
        "coverage", "run",
        "-m", "pytest",
        "tests/",
        "-v"
    ], cwd=base_dir)
    
    if result.returncode != 0:
        print("Tests failed!")
        sys.exit(1)
    
    # Generate reports
    print("\nGenerating coverage reports...")
    subprocess.run(["coverage", "report"], cwd=base_dir)
    subprocess.run(["coverage", "html"], cwd=base_dir)
    subprocess.run(["coverage", "xml"], cwd=base_dir)
    
    print("\nCoverage reports generated:")
    print(f"- HTML report: {base_dir}/coverage_html/index.html")
    print(f"- XML report: {base_dir}/coverage.xml")
    print("\nYou can open the HTML report in your browser to see detailed coverage information.")

if __name__ == "__main__":
    run_coverage() 