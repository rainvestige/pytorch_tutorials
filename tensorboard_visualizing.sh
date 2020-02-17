#! /bin/bash
echo "Make sure you are in conda virtual env"
rm -r runs
python ./tensorboard_visualizing.py && tensorboard --logdir=runs
echo "Script Finished"
