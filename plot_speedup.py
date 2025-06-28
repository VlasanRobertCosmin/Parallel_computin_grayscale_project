#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Remove seaborn import - not needed

def plot_speedup_graphs():
    """Generate comprehensive speedup visualization"""
    
    # Read the benchmark results
    try:
        df = pd.read_csv('speedup_results.csv')
        print("✅ CSV file loaded successfully")
    except FileNotFoundError:
        print("Error: speedup_results.csv not found. Run the benchmark first.")
        return
    
    # Extract algorithm and image size
    df[['Algorithm', 'ImageSize']] = df['Algorithm'].str.split('_', expand=True)
    df['ImageSize'] = df['ImageSize'].astype(int)
    
    # Set up the plotting style (use compatible styles)
    try:
        plt.style.use('seaborn')  # Try seaborn first
        print("Using seaborn style")
    except:
        try:
            plt.style.use('ggplot')  # Fallback to ggplot
            print("Using ggplot style")
        except:
            plt.style.use('default')  # Final fallback
            print("Using default style")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parallel Image Processing Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Speedup comparison by algorithm
    ax1 = axes[0, 0]
    algorithms = df['Algorithm'].unique()
    x = np.arange(len(algorithms))
    width = 0.35
    
    openmp_speedups = df.groupby('Algorithm')['OpenMP_Speedup'].mean()
    mpi_speedups = df.groupby('Algorithm')['MPI_Speedup'].mean()
    
    # Convert to numpy arrays to avoid pandas indexing issues
    openmp_values = openmp_speedups.values
    mpi_values = mpi_speedups.values
    
    bars1 = ax1.bar(x - width/2, openmp_values, width, label='OpenMP', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, mpi_values, width, label='MPI', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Average Speedup')
    ax1.set_title('Average Speedup by Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{openmp_values[i]:.2f}', ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mpi_values[i]:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Speedup vs Image Size
    ax2 = axes[0, 1]
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['Algorithm'] == algo].sort_values('ImageSize')
        # Convert pandas Series to numpy arrays to avoid indexing issues
        image_sizes = algo_data['ImageSize'].values
        openmp_speedups = algo_data['OpenMP_Speedup'].values
        mpi_speedups = algo_data['MPI_Speedup'].values
        
        ax2.plot(image_sizes, openmp_speedups, 
                color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                linestyle='-', label=f'{algo} (OpenMP)', linewidth=2, markersize=6)
        ax2.plot(image_sizes, mpi_speedups, 
                color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                linestyle='--', label=f'{algo} (MPI)', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_xlabel('Image Size (pixels)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Image Size')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Fix for older matplotlib versions - use basex instead of base
    try:
        ax2.set_xscale('log', base=2)
    except:
        ax2.set_xscale('log', basex=2)
    
    # 3. Execution Time Comparison (using matplotlib instead of seaborn)
    ax3 = axes[1, 0]
    
    # Focus on largest image size for clarity
    largest_size = df['ImageSize'].max()
    df_largest = df[df['ImageSize'] == largest_size]
    
    # Manual bar plot instead of seaborn
    algorithms_largest = df_largest['Algorithm'].unique()
    x_pos = np.arange(len(algorithms_largest))
    
    # Extract times for each algorithm
    seq_times = []
    omp_times = []
    mpi_times = []
    
    for algo in algorithms_largest:
        algo_row = df_largest[df_largest['Algorithm'] == algo].iloc[0]
        seq_times.append(algo_row['Sequential'])
        omp_times.append(algo_row['OpenMP'])
        mpi_times.append(algo_row['MPI'])
    
    width = 0.25
    ax3.bar(x_pos - width, seq_times, width, label='Sequential', alpha=0.8, color='gray')
    ax3.bar(x_pos, omp_times, width, label='OpenMP', alpha=0.8, color='skyblue')
    ax3.bar(x_pos + width, mpi_times, width, label='MPI', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title(f'Execution Time Comparison ({largest_size}x{largest_size} image)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms_largest, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency Analysis
    ax4 = axes[1, 1]
    
    # Calculate efficiency (assuming 4 cores/processes)
    num_cores = 4
    df['OpenMP_Efficiency'] = df['OpenMP_Speedup'] / num_cores * 100
    df['MPI_Efficiency'] = df['MPI_Speedup'] / num_cores * 100
    
    x = np.arange(len(algorithms))
    openmp_eff = df.groupby('Algorithm')['OpenMP_Efficiency'].mean()
    mpi_eff = df.groupby('Algorithm')['MPI_Efficiency'].mean()
    
    # Convert to numpy arrays
    openmp_eff_values = openmp_eff.values
    mpi_eff_values = mpi_eff.values
    
    bars1 = ax4.bar(x - width/2, openmp_eff_values, width, label='OpenMP', alpha=0.8, color='green')
    bars2 = ax4.bar(x + width/2, mpi_eff_values, width, label='MPI', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('Parallel Efficiency (4 cores/processes)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal (100%)')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{openmp_eff_values[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mpi_eff_values[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Speedup analysis saved to 'speedup_analysis.png'")
    
    # Print summary statistics
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Average OpenMP Speedup: {df['OpenMP_Speedup'].mean():.2f}x")
    print(f"Average MPI Speedup: {df['MPI_Speedup'].mean():.2f}x")
    print(f"Best OpenMP Performance: {df['OpenMP_Speedup'].max():.2f}x ({df.loc[df['OpenMP_Speedup'].idxmax(), 'Algorithm']})")
    print(f"Best MPI Performance: {df['MPI_Speedup'].max():.2f}x ({df.loc[df['MPI_Speedup'].idxmax(), 'Algorithm']})")
    
    # Efficiency analysis
    avg_openmp_eff = df['OpenMP_Efficiency'].mean()
    avg_mpi_eff = df['MPI_Efficiency'].mean()
    print(f"\nAverage OpenMP Efficiency: {avg_openmp_eff:.1f}%")
    print(f"Average MPI Efficiency: {avg_mpi_eff:.1f}%")
    
    if avg_openmp_eff > 75 and avg_mpi_eff > 75:
        print("✅ Both implementations show good parallel efficiency!")
    elif avg_openmp_eff > 75:
        print("✅ OpenMP shows good efficiency, MPI needs optimization")
    elif avg_mpi_eff > 75:
        print("✅ MPI shows good efficiency, OpenMP needs optimization")
    else:
        print("⚠️  Both implementations need optimization for better efficiency")

if __name__ == "__main__":
    plot_speedup_graphs()
