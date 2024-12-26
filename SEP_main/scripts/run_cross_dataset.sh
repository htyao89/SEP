#!/bin/bash

cd ..

# custom config
DATA=/hy-tmp/coop_data/ # path of dataset
TRAINER=SEP
WEIGHT=8.0 # weight of the textual consistency 
WEIGHT_V=6.0 # weight of the visual consistency

CFG=vit_cross_dataset
CTP=end  # class token position (end or middle)
NCTX=6 # length of textual prompts
NCTX_V=4 # length of visual prompts
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output
IND=0


for DATASET in imagenet
do
for SEED in 1 2 3
do
DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.COOP.N_CTX_V ${NCTX_V} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.L_IND ${IND} \
        TRAINER.COOP.W_V ${WEIGHT_V} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} 
    fi
done
done


for DATASET in eurosat dtd fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 caltech101 sun397
do
LOADEP=5
for SEED in 1 2 3
do
    COMMON_DIR_=imagenet/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${COMMON_DIR_}
    DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/test_${SUB}/${COMMON_DIR}

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
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.N_CTX_V ${NCTX_V} \
        TRAINER.COOP.L_IND ${IND} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} 
    fi
done
done
