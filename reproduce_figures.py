#!/usr/bin/env python3
"""
Main script to reproduce all figures from the paper.

Usage:
    python reproduce_figures.py                    # Generate all figures
    python reproduce_figures.py --figures 1 2 3    # Generate specific figures
    python reproduce_figures.py --output ./my_figs # Custom output directory

Prerequisites:
    1. Edit config_automate.py to set DATA_DIR to your data location
    2. Run setup_data.py to download and prepare all datasets
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add repo to path
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from config_automate import validate_setup, DATA_DIR
from figure_generators.fig1 import generate_fig1
from figure_generators.fig2 import generate_fig2
from figure_generators.fig3 import generate_fig3
from figure_generators.fig4 import generate_fig4ab, generate_fig4cd
from figure_generators.fig5 import generate_fig5


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce figures from the paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reproduce_figures.py                    # All figures
  python reproduce_figures.py --figures 1 2      # Only Fig 1 and 2
  python reproduce_figures.py --output ./figs    # Custom output location
        """
    )
    parser.add_argument(
        '--figures',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5],
        help='Specific figures to generate (e.g., --figures 1 2 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for figures (default: ./figure_outputs/YYYYMMDD_HHMM)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip setup validation check'
    )
    
    args = parser.parse_args()
    
    # Validate setup unless skipped
    if not args.skip_validation:
        print("=" * 70)
        print("Validating setup...")
        print("=" * 70)
        is_valid, missing = validate_setup()
        if not is_valid:
            print("\nERROR: Setup validation failed!")
            print("\nMissing paths:")
            for path in missing[:10]:
                print(f"  - {path}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            print("\nPlease ensure:")
            print("  1. config_automate.py has correct DATA_DIR")
            print("  2. setup_data.py has been run successfully")
            sys.exit(1)
        print("Setup validated successfully.\n")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = REPO_DIR / "figure_outputs" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Determine which figures to generate
    figures_to_generate = args.figures if args.figures else [1, 2, 3, 4, 5]
    
    print("=" * 70)
    print(f"Generating figures: {', '.join(map(str, figures_to_generate))}")
    print("=" * 70)
    print()
    
    results = {}
    
    # Generate figures
    for fig_num in figures_to_generate:
        print(f"\n{'=' * 70}")
        print(f"FIGURE {fig_num}")
        print(f"{'=' * 70}")
        
        try:
            if fig_num == 1:
                print("Generating Figure 1: Seed/Simulation/Experiment comparison & VAE reconstruction")
                paths = generate_fig1(output_dir)
                results[1] = {'status': 'success', 'paths': paths}
                print(f"Figure 1 complete: {len(paths)} files generated")
                
            elif fig_num == 2:
                print("Generating Figure 2: Seed to Intermediate pattern prediction")
                paths = generate_fig2(output_dir)
                results[2] = {'status': 'success', 'paths': paths}
                print(f"Figure 2 complete: {paths[0]}")
                
            elif fig_num == 3:
                print("Generating Figure 3: Intermediate to Complex pattern prediction")
                paths = generate_fig3(output_dir)
                results[3] = {'status': 'success', 'paths': paths}
                print(f"Figure 3 complete: {paths[0]}")
                
            elif fig_num == 4:
                print("Generating Figure 4a: Data efficiency (without augmentation)")
                paths_4a = generate_fig4ab(output_dir)
                print(f"Figure 4a complete: {len(paths_4a)} files generated")
                
                print("\nGenerating Figure 4b: Data efficiency (with augmentation)")
                paths_4b = generate_fig4cd(output_dir)
                print(f"Figure 4b complete: {len(paths_4b)} files generated")
                
                results[4] = {'status': 'success', 'paths': paths_4a + paths_4b}
                
            elif fig_num == 5:
                print("Generating Figure 5: Simulation to Experiment translation (ControlNet)")
                print("Note: This includes running inference, may take some time...")
                paths = generate_fig5(output_dir)
                results[5] = {'status': 'success', 'paths': paths}
                print(f"Figure 5 complete: {paths[0]}")
                
        except Exception as e:
            print(f"Error generating Figure {fig_num}: {e}")
            import traceback
            traceback.print_exc()
            results[fig_num] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = [k for k, v in results.items() if v['status'] == 'success']
    failed = [k for k, v in results.items() if v['status'] == 'failed']
    
    if successful:
        print(f"\nSuccessfully generated {len(successful)} figure(s): {', '.join(map(str, successful))}")
        print(f"\nAll outputs saved to: {output_dir}")
        
        print("\nGenerated files:")
        for fig_num in successful:
            for path in results[fig_num]['paths']:
                print(f"  - {Path(path).name}")
    
    if failed:
        print(f"\nFailed to generate {len(failed)} figure(s): {', '.join(map(str, failed))}")
        for fig_num in failed:
            print(f"  Figure {fig_num}: {results[fig_num]['error']}")
    
    print("\n" + "=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
