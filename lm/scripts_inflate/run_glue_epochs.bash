GPUS=$1
MODEL=$2
 
python3 /opt/tiger/inflation/lm/vlm/run_glue_epochs.py --gpus $GPUS --load $MODEL --snaps -1

