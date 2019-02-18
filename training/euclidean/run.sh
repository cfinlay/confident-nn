#!/bin/sh

# Shell script to launch PyTorch model testing experiments.
# This is a template to use as a starting point for creating
# your own script files. 

# Instructions for use:
# Make sure the paths are correct and execute from the command
# line using:
# $ ./yourscript.sh
# You will have to change file permissions before you can
# execute it:
# $ chmod +x yourscript.sh
# To automate the execution of multiple scipts use the
# jobdispatcher.py tool.

MODEL='AllCNN'

# Setup
TIMESTAMP=`date +%y-%m-%dT%H%M%S`  # Use this in LOGDIR
DATASET='cifar10'   # Use the dataset name in LOGDIR
DATADIR='/home/campus/oberman-lab/data/'  # Shared data file store

BASELOG='./logs/'$DATASET/$MODEL
LOGDIR=$BASELOG/'logsumexp-'$TIMESTAMP
SCRATCH='/mnt/data/scratch/runs/'$TIMESTAMP

mkdir -p $DATADIR
mkdir -p $SCRATCH
chmod g+rwx $SCRATCH
mkdir -p $BASELOG

ln -s $SCRATCH $LOGDIR


# If you want to specify which GPU to run on,
# prepend the following with
CUDA_VISIBLE_DEVICES=3 \
python -u ./train.py \
    --bn \
    --lr 1e-1 \
    --cutout 16 \
    --model $MODEL \
    --dataset $DATASET \
    --datadir $DATADIR \
    --logdir $LOGDIR \
    | tee $LOGDIR/log.out 2>&1 # Write stdout directly to log.out
                           # if you want to see results in real time,
                           # use tail -f

                           # If you don't want to see the output, replace
                           # '| tee' with '>'

rm $LOGDIR
mv $SCRATCH $LOGDIR
