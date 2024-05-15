# pkill -f "inference.py"
ps -ef | grep "inference.py" | grep -v grep | awk '{print $2}' | xargs kill -9

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch

nohup python inference.py 2>&1 > inference.log & 


# conda activate torch
# python webuis/character_manager/webui.py