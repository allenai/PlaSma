#!/bin/bash


NUM_RANDOM_SEEDS=1
LEARNINGRATES=( 1e-5 )  #( 1e-4 3e-4 1e-5 1e-6 )
BATCHSIZES=( 8 ) #( 8 16 32 )
EPOCHS=( 20 ) #( 10 20 30 )
MODELS=( t5-11b )


# script distillation
DATA_DIR="../data/CoPlan/planning"
OUT_DIR="./outputs"


for model in "${MODELS[@]}"
do
  export SEED=$((RANDOM))
  for bs in "${BATCHSIZES[@]}"
  do
    for lr in "${LEARNINGRATES[@]}"
    do
      export LEARNING_RATE=${l}
      for ep in "${EPOCHS[@]}"
      do
        mkdir -p ${OUT_DIR}/$(basename "$model")_bs${bs}_lr${lr}_ep${ep}_seed${SEED}
        cp $0 ${OUT_DIR}/$(basename "$model")_bs${bs}_lr${lr}_ep${ep}_seed${SEED}
        python distillation.py \
        --model_name_or_path $model \
        --do_train \
        --do_eval \
        --train_file ${DATA_DIR}/train.json \
        --validation_file ${DATA_DIR}/dev.json \
        --test_file $DATA_DIR/test.json \
        --source_prefix "provide steps:" \
        --evaluation_strategy "steps" \
        --metric_for_best_model 'eval_loss' \
        --save_total_limit 1 \
        --load_best_model_at_end True \
        --max_target_length 225 \
        --max_source_length 25 \
        --per_device_train_batch_size ${bs} \
        --gradient_accumulation_steps 16 \
        --per_device_eval_batch_size 16 \
        --adafactor \
        --learning_rate ${lr} \
        --warmup_ratio 0.08 \
        --num_train_epochs ${ep} \
        --seed ${SEED} \
        --save_strategy "steps" \
        --save_steps 200 \
        --eval_steps 200 \
        --logging_steps 200 \
        --overwrite_output_dir \
        --output_dir ${OUT_DIR}/$(basename "$model")_bs${bs}_lr${lr}_ep${ep}_seed${SEED} \
        --text_column "goal" \
        --summary_column "script" \
        --overwrite_cache \
        --predict_with_generate \
        --do_predict \
        --num_beams 5

      done
    done
  done
done
