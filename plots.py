import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('timing_results.csv', header=None, names=['type', 'count', 'time'])

# Get sequential time (assuming only one row)
seq_time = df[df['type'] == 'seq']['time'].values[0]

# Calculate speedup
df['speedup'] = seq_time / df['time']

# Filter data
omp = df[df['type'] == 'omp'].sort_values('count')
mpi = df[df['type'] == 'mpi'].sort_values('count')

# Plot combined speedup
plt.figure(figsize=(8, 6))
plt.plot(omp['count'].to_numpy(), omp['speedup'].to_numpy(), marker='o', linestyle='-', color='b', label='OpenMP')
plt.plot(mpi['count'].to_numpy(), mpi['speedup'].to_numpy(), marker='s', linestyle='-', color='g', label='MPI')

# Reference line: ideal speedup (linear)
max_count = max(omp['count'].max(), mpi['count'].max())
ideal_x = list(range(1, max_count + 1))
ideal_y = ideal_x  # perfect linear speedup
plt.plot(ideal_x, ideal_y, linestyle='--', color='gray', label='Ideal Linear Speedup')

# Formatting
plt.xlabel('Number of Threads / Processes')
plt.ylabel('Speedup (Sequential Time / Parallel Time)')
plt.title('Speedup Comparison: OpenMP vs MPI')
plt.xticks(sorted(set(omp['count']).union(mpi['count'])))
plt.ylim(0, max(max(omp['speedup'].max(), mpi['speedup'].max()) * 1.2, 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('combined_speedup.png')

plt.show()
