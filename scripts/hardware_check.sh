#!/bin/bash
# Step 0: run this and record the output. Then edit config.yaml (hardware section).
echo "=== Hardware ==="
system_profiler SPHardwareDataType 2>/dev/null | grep -E "Chip|Memory|Model Name"
echo "=== Free storage ==="
df -h /
echo "=== Python ==="
python3 --version
echo "=== MLX GPU check (run after pip install) ==="
python3 -c "import mlx.core as mx; print('MLX default device:', mx.default_device())" 2>/dev/null || echo "mlx not installed yet"
