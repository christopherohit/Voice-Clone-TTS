import wave
import contextlib
import os
from tqdm import tqdm
from mutagen.mp3 import MP3
from utils import get_tracking_file, checkLenght, split_audio, create_total_file
import glob
from moviepy.editor import *
import json
from data import text
import subprocess
        
        

model_file = []
model_file_aligner = []
model_file_pitch = []

list_all_audio = glob.glob('/vtca/nhanht/VTCA_viDataset/raw_wav/*.wav')

print("STARTING PROCESS FILE AUDIO ..................")
for audio in list_all_audio:
    fname = audio
    speaker_id = fname.split('/')[-1].split('_')[0]
    out_fname = f"/vtca/nhanht/VTCA_viDataset/wave/{speaker_id}"
    os.mkdir(out_fname)
    list_text, list_offset, list_dur, bool_variable = get_tracking_file(fname= fname)
    # if bool_variable is True:
    #     pass
    # else:
    #     list_text, list_offset, list_dur, bool_variable = get_tracking_file(fname= fname)
    list_condition_text, list_condition_start, list_condition_end = checkLenght(list_text= list_text,
                                                                                list_offset= list_offset,
                                                                                list_dur= list_dur) 
    
    for i, file in tqdm(enumerate(list_condition_text)):
        split_audio(in_fname= fname,
                    out_fname= f"{out_fname}/{str(int(list_condition_start[i])).zfill(5)}_{str(int(list_condition_end[i])).zfill(5)}.wav",
                    start= list_condition_start[i],
                    end= list_condition_end[i])
        cleaned_text = text._clean_text(file, cleaner_names=['vietnamese_cleaner'])
        
        model_file_aligner.append(f"{out_fname}/{str(int(list_condition_start[i])).zfill(5)}_{str(int(list_condition_end[i])).zfill(5)}.wav|{cleaned_text}|{speaker_id}")
        model_file_pitch.append(f"{out_fname}/{str(int(list_condition_start[i])).zfill(5)}_{str(int(list_condition_end[i])).zfill(5)}.wav|{speaker_id}")
        model_file.append(f"{out_fname}/{str(int(list_condition_start[i])).zfill(5)}_{str(int(list_condition_end[i])).zfill(5)}.wav|{cleaned_text}")
    
total_stage_file = [model_file_aligner, model_file_pitch, model_file]
create_total_file(out_path= '/vtca/nhanht/VTCA_viDataset',
                  list_file_txt= total_stage_file)
denoiserIT = input("Do you wanna denoise it ?(y/N)")
if denoiserIT == "N" or denoiserIT == "":
    print("PROCESSING COMPLETED ....................")
else:
    
    path = input('Please paste full directory of audio path: ')
    makeCopy = input('Make a copy it (Y/n):   ')
    
    assert path != None
    print("DENOISER IS PROCESSING ...................")
    if makeCopy == 'y' or makeCopy == '':
        subprocess.run(['python', '/home/nhan/voice-Research/viProcessing/denoiser_it.py',
                        '--path', f"{path}"])
    else:
        subprocess.run(['python', '/home/nhan/voice-Research/viProcessing/denoiser_it.py',
                        '--path', f"{path}",
                        '--makeCopy', False])