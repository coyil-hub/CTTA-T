#!/bin/bash

REPO=$PWD
MODEL=${1:-xlm-roberta}
TGT=${2:-mrqa}
GPU=${3:-0}
MODEL_PATH=${4}
SEED=${5:-42}
METHOD=${6:-ccc}
threshold=${7:-0.1}
consistency_threshold=${8:-1.0}
Lr=${9:-0.00001}
DATA_DIR=${10:-"$REPO/download/"}
OUT_DIR=${11:-"$REPO/continual_${METHOD}_outputs/"}

LR=$Lr
seed=$SEED
CURRENT_MODEL_PATH=$MODEL_PATH
t=$threshold
ct=$consistency_threshold

RESULTS_LOG="${OUT_DIR}results.csv"
echo "Dataset,EM,F1" > $RESULTS_LOG

langs=(noiseQA-syn.keyboard)

if [ $METHOD == "OIL" ]; then
  topk=0.5    #--> denoising
  m=0.99      #--> for expert update
  debias=1    #--> do debias or not
  ccc_restore=0
  cotta_restore=0
elif [ $METHOD == "cotta" ]; then
  m=0.99   #--> for expert update
  topk=0
  debias=0
  cotta_restore=1
  ccc_restore=0
elif [ $METHOD == "ccc" ]; then
  m=0.99     #--> for expert update
  topk=-1
  debias=0
  ccc_restore=1
  cotta_restore=0
elif [ $METHOD == "PL" ]; then
  topk=0.5    #--> denoising
  m=0    #--> for expert update
  debias=0    #--> do debias or not
  ccc_restore=0
  cotta_restore=0
elif [ $METHOD == "Tent" ]; then
  topk=-1    #--> denoising
  m=0    #--> for expert update
  debias=0    #--> do debias or not
  ccc_restore=0
  cotta_restore=0
fi

echo "Processing dataset: ${langs[@]}"

DIR=${DATA_DIR}/${TGT}/
TEST_FILE=${DIR}/${langs[0]}.json
PRED_DIR=${OUT_DIR}/${TGT}/
mkdir -p "${PRED_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_qa_tta.py \
  --model_type xlm-roberta \
  --model_name_or_path ${CURRENT_MODEL_PATH} \
  --initial_model_path ${MODEL_PATH}\
  --do_METHOD \
  --debias $debias \
  --topk $topk \
  --option 1 \
  --ccc_restore $ccc_restore \
  --cotta_restore $cotta_restore \
  --m $m \
  --alpha 1 \
  --seed $seed \
  --learning_rate $LR \
  --per_gpu_eval_batch_size 16 \
  --num_train_epochs 1 \
  --eval_lang ${langs[0]} \
  --predict_file "${TEST_FILE}" \
  --output_dir "${PRED_DIR}" \
  --option_max 1 \
  --threshold "${t}"\
  --consistency_threshold "${ct}"

PRED_FILE=${PRED_DIR}/predictions_${langs[0]}_.json
EVAL_SCRIPT=${PWD}/third_party/evaluate_squad.py
EVAL_RESULT=$(python "$EVAL_SCRIPT" "$TEST_FILE" "$PRED_FILE")
echo "${langs[0]},$EVAL_RESULT" >> $RESULTS_LOG

CURRENT_MODEL_PATH="${PRED_DIR}/checkpoint_last"


echo "All datasets processed. Results saved in $RESULTS_LOG"