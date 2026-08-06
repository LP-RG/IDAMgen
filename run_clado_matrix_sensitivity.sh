#!/bin/bash

BASE_DIR="tools/multi_layer_approx/sensitivity_matrix_generation/experiments/21x21_CLADO_method"
SCRIPT="tools/multi_layer_approx/sensitivity_matrix_generation/extract_sensitivity.py"

nohup python3 "$SCRIPT" \
  --1_1_approx "$BASE_DIR/1_1" \
  --1_2_approx "$BASE_DIR/1_2" \
  --2_1_approx "$BASE_DIR/2_1" \
  --2_2_approx "$BASE_DIR/2_2" \
  --2_s_approx "$BASE_DIR/2_s" \
  --3_1_approx "$BASE_DIR/3_1" \
  --3_2_approx "$BASE_DIR/3_2" &