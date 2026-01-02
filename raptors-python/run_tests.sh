#!/bin/bash
# Test runner script for Raptors Python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Raptors Python Test Suite ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if module is installed
echo -e "${YELLOW}Checking if raptors module is available...${NC}"
if python -c "import raptors" 2>/dev/null; then
    echo -e "${GREEN}✓ Module is installed${NC}"
else
    echo -e "${RED}✗ Module not found. Building with maturin...${NC}"
    if command -v maturin &> /dev/null; then
        maturin develop
        echo -e "${GREEN}✓ Module built successfully${NC}"
    else
        echo -e "${RED}✗ maturin not found. Please install it: pip install maturin${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${YELLOW}Running Rust unit tests...${NC}"
if cargo test --lib 2>&1 | tee /tmp/raptors_rust_tests.log; then
    echo -e "${GREEN}✓ Rust tests passed${NC}"
else
    echo -e "${RED}✗ Rust tests failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running Python pytest tests...${NC}"
if pytest tests/ -v; then
    echo -e "${GREEN}✓ Python tests passed${NC}"
else
    echo -e "${RED}✗ Python tests failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== All tests passed! ===${NC}"

