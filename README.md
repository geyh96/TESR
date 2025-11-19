# TESR
Code for the manuscript ``Transfer Learning through Enhanced Sufficient Representation:  Enriching Source Domain Knowledge with Target Data''

# Environment Setup
This section describes how to set up the environment and install all required packages for running the project.
#### Create environment
```
conda create -n TESR python=3.9.16 -y
```

#### Activate
```
conda activate TESR
```

#### Install dependencies
```
pip install -r requirements.txt
```

# Structure
## Simulations
├── Fig1/          # Code for Figure 1 in the paper           
├── Fig2/          # Code for Figure 2 in the paper            
├── Fig3/          # Code for Figure 3 in the paper              
├── FigS1/         # Code for Supplementary Experiment S1        
├── FigS2/         # Code for Supplementary Experiment S2          
├── FigS3/         # Code for Supplementary Experiment S3           
├── FigS4/         # Code for Supplementary Experiment S4            
└── Table1/        # Code for Table 1 in the paper        

## Folder Contents

Each folder contains the following files and subdirectories:

### Comparison Method Folders
- `Case_DDR/` - DDR method implementation
- `Case_DNN/` - DNN method implementation
- `Case_FT/` - FT method implementation
- `Case_TESR/` - TESR method implementation (proposed method)
- `Case_TransIRM/` - TransIRM method implementation

### Core Files
- `gen_data.py` - Data generation script
- `my_energy.py` - Distance covariance and energy distance implementation
- `my_model.py` - Model architecture and implementation
- `AmyGetResult.py` - Results extraction and analysis script
- `AMARun_ama.sh` - Master script to run all experiments


### Core Files in Each SubFolder
- `Case_Cor.py` - Code to reproduce the experiment
- `ARrun.sh/ARrun1.sh` - Master script to run all experiments for this method.

# Reference
Yeheng GE*, Xueyu ZHOU*, Jian HUANG.  Transfer Learning through Enhanced Sufficient Representation:  Enriching Source Domain Knowledge with Target Data. Manuscript. 
arXiv: https://arxiv.org/abs/2502.20414
(* denotes co-first authorship) 

# Development
The code repository is released by Xueyu ZHOU(xueyu.zhou@connect.polyu.hk) and Yeheng GE(geyh96@foxmail.com) for replication of the numerical results in the manuscript.

Dr.GE and Mr.ZHOU are from Prof. HUANG's research group in Hong Kong PolyU.
