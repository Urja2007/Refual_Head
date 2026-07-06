$ErrorActionPreference = "Stop"

Write-Host "Starting Full Refusal Head Pipeline (Native)..."
Write-Host "------------------------------------------------"

Write-Host "`n[1/6] Running Level 0 (Sanity Check)..."
python src/level0_sanity_check_native.py

Write-Host "`n[2/6] Running Level 1 (Head Identification)..."
python src/level1_head_identification_native.py

Write-Host "`n[3/6] Running Level 2 (Core Ablation)..."
python src/level2_core_ablation_native.py

Write-Host "`n[4/6] Running Level 3 (Control Battery)..."
python src/level3_control_battery_native.py

Write-Host "`n[5/6] Running Level 4 (Mechanism Attribution)..."
python src/level4_mechanism_attribution_native.py

Write-Host "`n[6/6] Running Level 5 (Benchmarking & Metrics)..."
python src/level5_benchmark_native.py

Write-Host "`nPipeline completed successfully!"
