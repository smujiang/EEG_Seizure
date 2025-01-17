# Automatic recognition of epileptic seizures in the EEG

SzCORE compatible reproduction of [Zhu et al., 'Automated Seizure Detection using Transformer Models on Multi-Channel EEGs', 2023](https://doi.org/10.1109/BHI58575.2023.10313440).



# Run the demo code from zhu

### Install the environment
``` bash
conda create --prefix /path/to/conda_envs/eeg
conda config --set env_prompt '({name}) '
conda config --append envs_dirs /path/to/conda_envs
conda activate eeg

conda install python=3.12.8
pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
pip install epilepsy2bids==0.0.7
pip install scipy==1.15.0
pip install numpy==1.26.0
```
### Run prediction code
``` bash
python zhu/src/zhu/main.py
```



