#!/bin/bash

echo "🧪 Testing Parallel Image Processing System"
echo "=========================================="

# Test 1: Check dependencies
echo "1. Checking dependencies..."

if command -v mpicxx >/dev/null 2>&1; then
    echo "   ✅ MPI compiler found"
else
    echo "   ❌ MPI compiler not found"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    echo "   ✅ Python 3 found"
else
    echo "   ❌ Python 3 not found"
    exit 1
fi

# Test 2: Check OpenMP support
echo "2. Checking OpenMP support..."
if echo | mpicxx -fopenmp -dM -E - | grep -q "OPENMP"; then
    echo "   ✅ OpenMP support detected"
else
    echo "   ❌ OpenMP support not found"
fi

# Test 3: Check if source file exists
echo "3. Checking source files..."
if [ -f "parallel_image_processor.cpp" ]; then
    echo "   ✅ Source file found"
else
    echo "   ❌ parallel_image_processor.cpp not found"
    echo "   Please make sure you have the C++ source file"
    exit 1
fi

# Test 4: Try compilation
echo "4. Testing compilation..."
make clean &>/dev/null
if make &>/dev/null; then
    echo "   ✅ Compilation successful"
    make clean &>/dev/null
else
    echo "   ❌ Compilation failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! System is ready."
echo "Run './run_benchmark.sh' to start the full benchmark."
