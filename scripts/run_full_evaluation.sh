#!/bin/bash
# Full evaluation pipeline for MM models
# Usage: ./run_full_evaluation.sh <scale> [seed]
# Example: ./run_full_evaluation.sh 10 42
# Example: ./run_full_evaluation.sh 100-sparse-75 42

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale> [seed]"
    echo "Examples:"
    echo "  $0 10          # Uses default seed 42"
    echo "  $0 100 123     # Uses seed 123"
    echo "  $0 100-sparse-75 42"
    exit 1
fi

SCALE=$1
SEED=${2:-42}  # Default to seed 42 if not provided

# Handle sparse models
if [[ $SCALE == *"sparse"* ]]; then
    DATA_DIR="data/mm${SCALE}"
else
    DATA_DIR="data/mm${SCALE}"
fi

BASE_DIR="trainings/MM/MM-${SCALE}"
MODEL_DIR="${BASE_DIR}/MM-${SCALE}-${SEED}"

# Check if model exists
if [ ! -f "${MODEL_DIR}/ckpt.pt" ]; then
    echo "❌ Error: Model checkpoint not found at ${MODEL_DIR}/ckpt.pt"
    echo "Please train the model first with: ./train_mm.sh ${SCALE} ${SEED}"
    exit 1
fi

echo "🔬 Running full evaluation pipeline for MM-${SCALE} (seed ${SEED})"
echo "============================================"

# 1. Basic evaluation
echo "📊 Step 1: Basic evaluation (metrics 1-3)"
python scripts/mm_eval.py ${SCALE} --seeds ${SEED} > ${BASE_DIR}/eval_summary.txt
cat ${BASE_DIR}/eval_summary.txt

# 2. Advanced metrics (4 & 5)
echo -e "\n📈 Step 2: Advanced metrics (transition matrix & Markov property)"
python scripts/mm_eval_transition_matrix.py \
    --transformer_path ${MODEL_DIR}/ckpt.pt \
    --formal_model_path ${DATA_DIR}/model.pkl \
    --output ${MODEL_DIR}/mm_metrics_4_5.json

# 3. Calculate theoretical comparison
echo -e "\n🎯 Step 3: Theoretical comparison"
python calculate_theoretical_minimums.py | grep -A3 "MM-${SCALE}:"

# 4. Aggregate metrics
echo -e "\n📊 Step 4: Aggregating all metrics"
python scripts/aggregate_metrics.py \
    --base_dir ${BASE_DIR} \
    --output ${BASE_DIR}/comprehensive_metrics.csv

# 5. Generate summary report
echo -e "\n📝 Step 5: Generating evaluation report"
cat > ${BASE_DIR}/automated_evaluation_report.md << EOF
# MM-${SCALE} Automated Evaluation Report
Generated: $(date)

## Basic Metrics
$(grep -E "(Perplexity|Accuracy)" ${BASE_DIR}/eval_summary.txt)

## Theoretical Comparison
$(python calculate_theoretical_minimums.py | grep -A3 "MM-${SCALE}:")

## Advanced Metrics
- Transition Matrix Fidelity: $(python -c "import json; data=json.load(open('${MODEL_DIR}/mm_metrics_4_5.json')); print(f\"KL={data['mean_kl_divergence']:.6f} ({data['performance_level']})\") ")
- Markov Property: $(python -c "import json; data=json.load(open('${MODEL_DIR}/mm_metrics_4_5.json')); print('✅ PASS' if data['markov_property_results']['success'] else '❌ FAIL')")

## Files Generated
- Basic evaluation: ${BASE_DIR}/eval_summary.txt
- Advanced metrics: ${MODEL_DIR}/mm_metrics_4_5.json  
- Comprehensive CSV: ${BASE_DIR}/comprehensive_metrics.csv
- This report: ${BASE_DIR}/automated_evaluation_report.md
EOF

echo -e "\n✅ Full evaluation complete!"
echo "📄 Report saved to: ${BASE_DIR}/automated_evaluation_report.md"
cat ${BASE_DIR}/automated_evaluation_report.md