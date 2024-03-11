# PPFNet
This repository contains official implementaion of the paper "PPFNet: A Privacy Preserving Federated Network for Cervical Cancer Detection from PaP Smear Images".
Steps to run the algorithms:
1) run Data_creation.py after changing the data distribution you want.
2) run FedBN.py i.e. without any differential privacy(DP). It will load data and run data_augmentation_(1).py to use normal and advance augmentaion to the data, then runs the algorithm.
3) run FedBN+DP.py for DP. It contains both Adaptive differntial privacy(ADP) and only DP. uncomment the part you want.  It will load data and run data_augmentation_(1).py to use normal and advance augmentaion to the data, then runs the algorithm.
4) After training with all 3 models, run the Ensemling.py.
