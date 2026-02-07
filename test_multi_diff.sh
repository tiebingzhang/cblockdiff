#!/bin/bash
set -e

DATA_DIR="/mnt/data"
cd "$DATA_DIR"

# Clean up temporary files
rm -f f2.data f3.data f2.bdiff f3.bdiff f3-merge.data f3-diff.bdiff

echo "=== Step 1: Create f2.data (f1 + extra1) ==="
cp f1.data f2.data
cat extra1.data >> f2.data
echo "f2.data created"

echo "=== Step 2: Create first blockdiff (f1 -> f2) ==="
blockdiff create f2.bdiff f2.data --base f1.data
ls -lh f2.bdiff

echo "=== Step 3: Create f3.data (f2 + extra2) ==="
cp f2.data f3.data
cat extra2.data >> f3.data
echo "f3.data created"

echo "=== Step 4: Create second blockdiff (f2 -> f3) ==="
blockdiff create f3.bdiff f3.data --base f2.data
ls -lh f3.bdiff

echo "=== Step 5: Apply both diffs to f1 to create f3-merge.data ==="
blockdiff apply f3.bdiff f3-merge.data --base f1.data,f2.bdiff

echo "=== Step 6: Verify f3-merge.data matches f3.data ==="
sha1sum f3.data f3-merge.data
if [ "$(sha1sum f3.data | awk '{print $1}')" = "$(sha1sum f3-merge.data | awk '{print $1}')" ]; then
    echo "✓ MERGE TEST PASSED: f3-merge.data matches f3.data"
else
    echo "✗ MERGE TEST FAILED"
    exit 1
fi

echo "=== Step 7: Create diff using base f1 + f2.bdiff -> f3 ==="
blockdiff create f3-diff.bdiff f3.data --base f1.data,f2.bdiff

echo "=== Step 8: Verify f3-diff.bdiff matches f3.bdiff ==="
sha1sum f3.bdiff f3-diff.bdiff
if [ "$(sha1sum f3.bdiff | awk '{print $1}')" = "$(sha1sum f3-diff.bdiff | awk '{print $1}')" ]; then
    echo "✓ DIFF TEST PASSED: f3-diff.bdiff matches f3.bdiff"
else
    echo "✗ DIFF TEST FAILED"
    exit 1
fi

echo ""
echo "=== ALL TESTS PASSED ==="
