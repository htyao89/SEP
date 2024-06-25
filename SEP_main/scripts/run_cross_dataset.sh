#!/bin/bash

# cd ..

# custom config
DATA=/hy-tmp/data
TRAINER=TCP
WEIGHT=4.0
DATASET=imagenet
CFG=vit_b128_ep5_ctxv1_cross_dataset
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
L=0

for SEED in 1 2 3
do
    DIR=output_0617_cd/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.L ${L} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r eurosat dtd fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 caltech101 sun397
do
for SEED in 1 2 3
do
DIR=output_0617_cd/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output_0617_cd/base2new/train_base/imagenet/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED} \
    --load-epoch 5 \
    --eval-only \
    TRAINER.COOP.L ${L} \
    TRAINER.COOP.N_CTX 4
fi
done
done


