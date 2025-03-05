# OBM
A package for feature-based optimal biomarker selection and prediction

Installation:

pip install -r requirements_1.txt
pip install -r requirements_2.txt
pip3 install torch==1.13.0
pip3 install torchvision==0.14.0
pip3 install torchaudio==0.13.0
pip install tmux




Run program:

1. feature and class preparation 
  fobm-main/data/
  		LA_ID_12M_cont_0pad_PlusBasic_features.csv  # features used in OBM
  		LA_ID_12M_cont_0pad_PlusBasic_labels.csv # Subject IDs and group information
  
2. bash run.sh
