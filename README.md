# PPFNet
This repository contains official implementaion of the paper "Privacy Preserving Federated Learning for Cervical Cancer Detection using a Power-weighted Ensemble Technique".
Steps to run the algorithms:
1) run Data_creation.py after changing the data distribution you want.
2) run Proposed method without DP.py i.e. without any differential privacy(DP). It will load data and run data_augmentation.py to use normal and advance augmentaion to the data, then runs the algorithm.
3) run Proposed method.py to simulate the proposed method. It contains both Adaptive differntial privacy(ADP) and only DP. uncomment the part you want.  It will load data and run data_augmentation.py to use normal and advance augmentaion to the data, then runs the algorithm.
4) After training with all 3 models, run the Ensemling.py.
5) for t-SNE visulization, McNemar's Test and Conformal Prediction, run corresponding .py files.
6) To run the baseline algorithm i.e. Fedavg.py, run baseline.py. 
