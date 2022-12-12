# Robots Movement Classification
This tool can be used to determine the position and orientation direction based on the information provided by sensors.

## Mission
Mission of our software was to predict the movement of the user's robot, so users can get the location of their robots, or even design a route for robots.

## Project objective
The project is to use the ultrasound sensor data or distance reading data of the wall-followed robot to train a model. The model can be used to determine the position and orientation direction based on the data analyzed. We will try different machine learning algorithm and find out a model that has better prediction.

This software could be used in some of the storage. Like shipping storage, there will be some of the automotive vehicles to deliver with the setting path. Based on the data collection, customers could have the direction printed out to determine if the automotive vehicle followed the orders or not.
 
Similarly, any industry that uses a robot running in some fixed routes, can locate their robots depend on our software.

## Repository Structure
 ```
.
├── DATA
│   ├── sensor_readings_24.csv
│   └── sensor_readings_4.csv
├── doc
│   └── presentation.ipynb
├── draft_code
│   ├── SVM_GaussianGNB_data24.ipynb
│   ├── SVM_GaussianGNB_data4.ipynb
│   └── neuralnet_lineaerclassifer.py
├── src
│   └── Classifiers.py
├── LICENSE
├── README.md
├── RMC_ENV_SETUP.bash
└── env_RMC.yml
 ```

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
