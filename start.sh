pkill -f "app.py"
# ps -ef | grep "app.py" | awk '{print $2}' | xargs kill -9

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch

nohup python app.py 2>&1 > app.log & 


# conda activate torch
# python webuis/character_manager/webui.py