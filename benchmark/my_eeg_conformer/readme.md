

# Update log 03/06/2025
Sample a time window from the EEG signals. Using CNN models to achieve Seizure/non-seizure detection. 
## create training dataset
1. Check the code within [my_eeg_dataset.py](benchmark/my_eeg_conformer/my_eeg_dataset.py)

## model design
1. Check the code within [my_models.py](benchmark/my_eeg_conformer/my_models.py)

## model training
1. Check the code within [main.py](benchmark/my_eeg_conformer/main.py)

## TODO:
1. Add loss monitoring, set conditions to stop and save the trained model
2. Add testing code. Load the pretrained model and test a edf file.
3. Add results visualization, check the waveforms that are detected positive, show before and after 5 seconds as well
4. Create docker. Check performance using standard SzCORE pipeline.
5. Modify the dataset, balance the training set. Check the performance again.
6. Add transformer in the the architecture. Check the performance again.
5. 









