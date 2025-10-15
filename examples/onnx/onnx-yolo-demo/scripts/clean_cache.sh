#!/bin/bash
# Clean Python bytecode cache
# Run this if you encounter dtype or import issues after updating code

echo "Cleaning Python cache..."

# Get the root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Cleaning cache in: $REPO_ROOT"

# Delete all __pycache__ directories
echo "Removing __pycache__ directories..."
find "$REPO_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Delete all .pyc files
echo "Removing .pyc files..."
find "$REPO_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true

# Delete all .pyo files
echo "Removing .pyo files..."
find "$REPO_ROOT" -type f -name "*.pyo" -delete 2>/dev/null || true

echo "✓ Cache cleaned successfully!"
echo ""
echo "You can now run setup.sh or train.py with fresh imports."
