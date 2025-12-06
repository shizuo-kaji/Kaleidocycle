"""Demo: Kaleidocycle Report with Curvature Recursion

This example demonstrates how to generate a report for a kaleidocycle
that includes all geometric, topological, and constraint information,
including the newly added curvature recursion statistics.

Usage:
    python examples/report_demo.py
"""

from kaleidocycle import Kaleidocycle, random_hinges


def demo_optimized_kaleidocycle():
    """Show report for an optimized kaleidocycle."""
    print("=" * 60)
    print("Optimized Kaleidocycle Report")
    print("=" * 60)
    
    # Create optimized kaleidocycle
    kc = Kaleidocycle(6, seed=42)
    
    # Generate and print report
    print(kc.report())


def demo_random_kaleidocycle():
    """Show report for a random (non-optimized) kaleidocycle."""
    print("\n" + "=" * 60)
    print("Random Kaleidocycle Report")
    print("=" * 60)
    
    # Create from random hinges (not optimized)
    hinges = random_hinges(6, seed=123).as_array()
    kc = Kaleidocycle(hinges=hinges)
    
    # Generate and print report
    print(kc.report())


def demo_report_with_precision():
    """Show report with custom precision."""
    print("\n" + "=" * 60)
    print("Report with Custom Precision")
    print("=" * 60)
    
    kc = Kaleidocycle(8, oriented=True, seed=42)
    
    # Generate report with 3 decimal places
    print(kc.report(precision=3))


def main():
    """Run all demos."""
    demo_optimized_kaleidocycle()
    demo_random_kaleidocycle()
    demo_report_with_precision()


if __name__ == "__main__":
    main()
