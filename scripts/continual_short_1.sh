#!/bin/bash

REPO=$PWD
MODEL=${1:-xlm-roberta}
TGT=${2:-tasks}
GPU=${3:-0}
MODEL_PATH=${4}
METHOD=${5:-ccc}
threshold=${6:-0.1}
contribution_threshold=${7:-1.0}
Lr=${8:-0.00001}
DATA_DIR=${9:-"$REPO/download/"}
OUT_DIR=${10:-"$REPO/continual_${METHOD}_outputs/"}

SEED=$(shuf -i 1-10000 -n 1)
LR=$Lr
seed=$SEED
CURRENT_MODEL_PATH=$MODEL_PATH
t=$threshold
ct=$contribution_threshold

RESULTS_LOG="${OUT_DIR}results.csv"
echo "Dataset,EM,F1" > $RESULTS_LOG


for option in 1 2 3 4 5 6 7 ; do
  if [ $option == 1 ]; then
    langs=(syn.keyboard)
  elif [ $option == 2 ]; then
    langs=(syn.asr)
  elif [ $option == 3 ]; then
    langs=(syn.translation)
  elif [ $option == 4 ]; then
    langs=(Search)
  elif [ $option == 5 ]; then
    langs=(zh)
  elif [ $option == 6 ]; then
    langs=(ar)
  elif [ $option == 7 ]; then
    langs=(es)
  fi

  if [ $METHOD == "OIL" ]; then
    topk=0.5   
    m=0.99     
    debias=1   
    ccc_restore=0
    cotta_restore=0
  elif [ $METHOD == "cotta" ]; then
    m=0.99  
    topk=0
    debias=0
    cotta_restore=1
    ccc_restore=0
  elif [ $METHOD == "CTTA-T" ]; then
    m=0.99   
    topk=-1
    debias=0
    ccc_restore=1
    cotta_restore=0
  elif [ $METHOD == "PL" ]; then
    topk=0.5   
    m=0  
    debias=0   
    ccc_restore=0
    cotta_restore=0
  elif [ $METHOD == "Tent" ]; then
    topk=-1   
    m=0   
    debias=0   
    ccc_restore=0
    cotta_restore=0
  fi

  echo "Processing dataset: ${langs[@]}"

  DIR=${DATA_DIR}/${TGT}/
  TEST_FILE=${DIR}/${langs[0]}.json
  PRED_DIR=${OUT_DIR}/${TGT}/
  mkdir -p "${PRED_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_ctta_t.py \
    --model_type xlm-roberta \
    --model_name_or_path ${CURRENT_MODEL_PATH} \
    --initial_model_path ${MODEL_PATH}\
    --do_METHOD \
    --debias $debias \
    --memory_size 1 \
    --topk $topk \
    --option $option \
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
    --option_max 7 \
    --threshold "${t}"\
    --contribution_threshold "${ct}"


  PRED_FILE=${PRED_DIR}/predictions_${langs[0]}_.json
  EVAL_SCRIPT=${PWD}/third_party/evaluate_squad.py
  EVAL_RESULT=$(python "$EVAL_SCRIPT" "$TEST_FILE" "$PRED_FILE")
  echo "${langs[0]},$EVAL_RESULT" >> $RESULTS_LOG

  CURRENT_MODEL_PATH="${PRED_DIR}/checkpoint_last"
done

echo "All datasets processed. Results saved in $RESULTS_LOG"
