#!/bin/bash
# SSM Full Pipeline Example (4 Steps)
# 원본 SubDyve의 mine_subgraph.sh와 동일한 워크플로우

set -e

# Configuration
# 원본 SubDyve의 mine_subgraph.sh와 동일한 타겟명 사용
TARGETS=(
    "ACES"  # Acetylcholinesterase
    "ADA"   # Adenosine deaminase
    "EGFR"  # Epidermal growth factor receptor
)
THRESHOLD=0.9
TOP=${1:-100}  # default 100 (원본은 2000이지만 예제는 작게)

echo "=========================================="
echo "SSM Full Pipeline Example"
echo "Original SubDyve targets: ${TARGETS[@]}"
echo "Threshold: $THRESHOLD"
echo "Top-N: $TOP"
echo "=========================================="

# 이전 실행 결과 정리
echo ""
echo "[0/4] Cleaning previous results..."
rm -rf results/ logs/ data/
mkdir -p logs

# 예제 데이터 준비 (미리 생성된 CSV 파일 사용)
echo ""
echo "Preparing example data..."
for target in "${TARGETS[@]}"; do
    mkdir -p data/${target}

    # 미리 생성된 샘플 데이터 복사
    if [ ! -f "data/${target}/train.csv" ]; then
        if [ -f "examples/sample_targets/${target}/train.csv" ]; then
            cp examples/sample_targets/${target}/train.csv data/${target}/
            echo "  Copied examples/sample_targets/${target}/train.csv"
        else
            echo "  Warning: examples/sample_targets/${target}/train.csv not found"
            exit 1
        fi
    fi

    if [ ! -f "data/${target}/test.csv" ]; then
        if [ -f "examples/sample_targets/${target}/test.csv" ]; then
            cp examples/sample_targets/${target}/test.csv data/${target}/
            echo "  Copied examples/sample_targets/${target}/test.csv"
        else
            echo "  Warning: examples/sample_targets/${target}/test.csv not found"
            exit 1
        fi
    fi
done

# ==========================================
# Step 1: Run basic SSM on all targets
# ==========================================
echo ""
echo "[1/4] Step 1: Running SSM on all targets..."
for target in "${TARGETS[@]}"; do
    echo "  Running SSM for target: $target"

    train_data="data/${target}/train.csv"
    test_data="data/${target}/test.csv"
    output_dir="results/${target}/"
    mkdir -p "$output_dir"

    python src/ssm_smiles.py \
        --train_data "$train_data" \
        --test_data "$test_data" \
        --output_dir "$output_dir" \
        --rw 7 --alpha 0.1 --iterations 5 --nWalker 5 \
        --DiSC 1 --n_jobs 2
done

echo "✓ Step 1 completed."

# ==========================================
# Step 2: Collect best AUC iteration
# ==========================================
echo ""
echo "[2/4] Step 2: Aggregating best AUC iterations..."
mkdir -p logs

python src/collect.py \
    --threshold "$THRESHOLD" \
    --targets "${TARGETS[@]}"

echo "✓ Step 2 completed."
cat logs/target_max_AUC_dict_${THRESHOLD}.csv

# ==========================================
# Step 3: Run DiSC refinement
# ==========================================
echo ""
echo "[3/4] Step 3: Running DiSC refinement with best iterations..."
seed_max_AUC_dict="logs/target_max_AUC_dict_${THRESHOLD}.csv"

for target in "${TARGETS[@]}"; do
    echo "  Running DiSC refinement for: $target"

    # Read best iteration for this target
    if [ -f "$seed_max_AUC_dict" ]; then
        bestIteration=$(awk -F, -v seed="$target" '$1 == seed {print $2}' ${seed_max_AUC_dict})
        echo "    Best iteration: $bestIteration"
    else
        echo "    Warning: No AUC dict found, skipping"
        continue
    fi

    train_data="data/${target}/train.csv"
    test_data="data/${target}/test.csv"
    output_dir="results/${target}/"

    # Step 3에서는 ssm_DISC.py 사용 (원본과 동일)
    python src/ssm_DISC.py \
        --train_data "$train_data" \
        --test_data "$test_data" \
        --output_dir "$output_dir" \
        --rw 7 --alpha 0.1 --iterations 5 \
        --nWalker 5 --DiSC 3 \
        --bestIteration "$bestIteration"
done

echo "✓ Step 3 completed."

# ==========================================
# Step 4: Extract top-N subgraphs
# ==========================================
echo ""
echo "[4/4] Step 4: Extracting top-$TOP discriminative subgraphs..."

for target in "${TARGETS[@]}"; do
    echo "  Processing DISC extraction for: $target"
    python src/clean_DISC.py \
        --target "$target" \
        --top "$TOP" \
        --threshold "$THRESHOLD"
done

echo "✓ Step 4 completed."

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "✓ All steps completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
for target in "${TARGETS[@]}"; do
    echo "  - results/${target}/"
    echo "    - performance.csv"
    echo "    - iteration_*/DiSC_3.csv"
    echo "    - top_${TOP}_subgraphs.csv (from Step 4)"
done
echo ""
echo "Logs:"
echo "  - logs/target_max_AUC_dict_${THRESHOLD}.csv"
echo ""
echo "To run with different top-N: bash examples/pipeline_example.sh 2000"
