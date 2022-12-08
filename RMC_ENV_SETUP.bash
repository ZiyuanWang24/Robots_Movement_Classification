echo Create a enviroment 'env_RMC'
conda create -n env_RMC python=3.9
echo Update the env by env_RMC.yml
conda env update -n env_RMC -f env_RMC.yml