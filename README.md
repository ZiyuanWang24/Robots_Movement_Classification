# Robots Movement Classification
This tool can be used to determine the position and orientation direction based on the information provided by sensors.

## Project objective
The project is to use the ultrasound sensor data or distance reading data of the wall-followed robot to train a model. The model can be used to determine the position and orientation direction based on the data analyzed. We will try different machine learning algorithm and find out a model that has better prediction.

## Enviroment Setup
```
conda create -n env_RMC python=3.9
conda env update -n env_RMC -f env_RMC.yml
conda activate env_RMC
```
Or
```
bash RMC_ENV_SETUP.bash
conda activate env_RMC
```
