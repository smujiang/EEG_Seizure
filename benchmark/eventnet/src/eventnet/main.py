import torch
import numpy as np
import glob, os
import natsort
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from utils import load_model, get_dataloader, predict

def main(edf_file, outFile):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)

    assert eeg.montage is Eeg.Montage.UNIPOLAR, "Error: Only unipolar montages are supported."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model() # load model
    model.to(device)

    recording_duration = eeg.data.shape[1] / eeg.fs
    window_size_sec = 120
    if recording_duration-5 < window_size_sec * 2:
        win_size_sec = (recording_duration-5) // 20 * 10
    else:
        win_size_sec = window_size_sec

    if win_size_sec > 25:
        dataloader = get_dataloader(data=eeg.data, window_size_sec=win_size_sec, fs=eeg.fs)

        y_predict = predict(model=model, dataloader=dataloader, device=device, window_size_sec=win_size_sec, 
                            recording_duration=recording_duration, fs=eeg.fs, thresh=0.44)
    else:
        y_predict = np.zeros(int(recording_duration*eeg.fs))
    hyp = Annotations.loadMask(y_predict, eeg.fs)
    hyp.saveTsv(outFile)


if __name__ == "__main__":
    # data_dir = "/Users/jjiang10/Data/EEG/BIDS_Siena"
    data_dir = "/data/jjiang10/Data/EEG/BIDS_Siena"
    edf_file_list = glob.glob(data_dir + "/sub-*/ses-*/eeg/*.edf")
    for edf_file in natsort.natsorted(edf_file_list):
        print("Processing %s" % edf_file)
        outFile = edf_file.replace("BIDS_Siena", "BIDS_Siena_event_pred").replace(".edf", ".tsv")
        if os.path.exists(outFile):
            print("Already exists. Skip.")
        else:
            dir_ele = os.path.split(outFile)
            if not os.path.exists(dir_ele[0]):
                os.makedirs(dir_ele[0])
            main(edf_file, outFile)
    print("Done")

