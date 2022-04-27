# !/bin/bash

PROJ_PATH=/home/hminle/github/UPFD_ATT

# echo "Begin: " `date +'%Y-%m-%d %H:%M'`
# echo "Run: RESTAURANT"

# python $PROJ_PATH/src/main.py

echo "Begin: " `date +'%Y-%m-%d %H:%M'`

echo "Run: gossipcop"
# python $PROJ_PATH/src/main.py -config_file ../src/config/gnn_att_gos.json > ../log/gnn_att_gos.txt 2>&1
python $PROJ_PATH/src/main.py -config_file ../src/config/gnn_gos.json > ../log/gnn_gos.txt 2>&1

echo "Run: politifact"
# python $PROJ_PATH/src/main.py -config_file ../src/config/gnn_att_pol.json > ../log/gnn_att_pol.txt 2>&1
# python $PROJ_PATH/src/main.py -config_file ../src/config/gnn_pol.json > ../log/gnn_pol.txt 2>&1
