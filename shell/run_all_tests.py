import os
import subprocess
import sys
import webbrowser


def run_tests():
    """Run pytest with coverage and generate the report."""
    try:
        # Run pytest with coverage
        subprocess.run(['pytest', '--cov=src/tsroots', '--cov-report=term-missing'], check=True)

        # Generate the HTML coverage report
        subprocess.run(['coverage', 'html'], check=True)

        # Open the HTML coverage report in the default web browser
        report_path = os.path.abspath("htmlcov/index.html")
        webbrowser.open(f"file://{report_path}")

        print("Tests completed successfully. HTML coverage report opened in browser.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
