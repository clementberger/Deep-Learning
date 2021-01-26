# Deep-Learning
Project Deep Learning BERGER Clément, ER-RAMMACH Ilyes

# Installation
To use our code, it is strongly advised to create a dedicated environment with
```
conda env create -f environment.yml
conda activate dlproj
```

# Usage
The attention mechanism yelds some warnings, we advise you to call the code as follows :
```
python -W ignore code.py
```
The datasets were too big to be put on github. If you want to run our code, please email one of us so that we can send you what's required. If you do so, here is an example of code you can launch :
```
python -W ignore train_attention_multiruns.py --num_epochs 200 --batch_size 32 --id example --loss_type BCE --lr 0.0005 --log_dir ./log/CNN_all/ --model_type CNN --flip True --interpolation True --noise 0.2 --train ./data/train_scalo.npz --test ./data/test_scalo.npz 
```

# Acknowlegment
We used the OpenMIC dataset, from which we also took some code (especially the VGGish part)
https://github.com/cosmir/openmic-2018

For the attention part, we were inspired by https://github.com/SiddGururani/AttentionMIC
