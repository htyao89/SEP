#!/bin/bash

# cd ..

# custom config
DATA=/hy-tmp/data
TRAINER=TCP
WEIGHT=4.0
CFG=vit_b32_ep50_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=4  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output_0625_fsl
L=0

# for DATASET in eurosat dtd fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 caltech101
# do
# for SEED in 1 2 3
# do
#     DIR=${FOLDER}_${NCTX}_${L}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
#     if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}. Skip this job"
#     else
#         echo "Run this job and save the output to ${DIR}"
#         python train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --output-dir ${DIR} \
#         TRAINER.COOP.N_CTX ${NCTX} \
#         TRAINER.COOP.CSC ${CSC} \
#         TRAINER.COOP.W ${WEIGHT} \
#         TRAINER.COOP.L ${L} \
#         TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
#         DATASET.NUM_SHOTS ${SHOTS} 
#     fi
# done
# done


CFG=vit_b128_ep10_ctxv1
for DATASET in imagenet sun397
do
for SEED in 1 2 3
do
    DIR=${FOLDER}_${NCTX}_${L}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
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
done
