#!/bin/bash

cd ..

# custom config
DATA=/hy-tmp/coop_data/ # path of dataset
TRAINER=SEP
WEIGHT=8.0 # weight of the textual consistency 
WEIGHT_V=6.0 # weight of the visual consistency

CFG=vit_b32_ep50_ctxv1
CTP=end  # class token position (end or middle)
NCTX=6 # length of textual prompts
NCTX_V=4 # length of visual prompts
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output
IND=0 # indext of selected layers. 0: [1,2,3,4,5,6,7,8,9,10,11]

for NCTX in 6
do
for NCTX_V in 4
do
for DATASET in eurosat dtd fgvc_aircraft ucf101 oxford_pets fgvc_aircraft stanford_cars eurosat  oxford_flowers oxford_pets stanford_cars ucf101 caltech101 food101
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
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi

    LOADEP=50
    SUB=new
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${COMMON_DIR}
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
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
done

CFG=vit_b128_ep10_ctxv1

for DATASET in sun397
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
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=10
SUB=new
for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${COMMON_DIR}
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
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
done
done
done
