import os
from model_denoiser.getText import stt
import wave
import contextlib
from tqdm import tqdm
from pydub import AudioSegment



def get_tracking_file(fname):
    
    request_tracking_file = f"{fname.split('/')[-1].split('.')[0]}.txt"
    all_file_tracking = os.listdir('/home/nhan/voice-Research/viProcessing/tracking_raw/')
    if request_tracking_file in all_file_tracking:
        list_text, list_offset, list_dur = [], [], []
        with open(f'/home/nhan/voice-Research/viProcessing/tracking_raw/{request_tracking_file}', 'r') as fp:
            to_list = fp.readlines()
            
        for line in to_list:
            if line.split('|')[0] == "":
                pass
            else:
                list_text.append(line.split('|')[0])
                list_offset.append(line.split("|")[1])
                list_dur.append(line.split('|')[-1].replace('\n', ''))
        isComplete = True
        return list_text, list_offset, list_dur, isComplete
    else:
        print("YOUR AUDIO HASN'T RUN SPEECH TO TEXT SO WE NOT FOUND LOG TRACKING FILE")
        print(f"RUNNING TRACKING {fname} AUDIO FILE")
        stt(input_file= fname, 
            out_file= f"/home/nhan/voice-Research/viProcessing/tracking_raw/{fname.split('/')[-1].split('.')[0]}.txt")
        isComplete = False
        
        return isComplete
    
    
def get_dur(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

    
def checkLenght(list_text, list_offset, list_dur):
    list_condition_text, list_condition_start, list_condition_end = [], [], []
    for i, text in enumerate(list_text):
        if (int(list_dur[i]) / 10000000) < 3:
            pass
        elif (int(list_dur[i]) / 10000000) >= 3:
            start = (int(list_offset[i]) / 10000000)
            end = start + ((int(list_dur[i]) / 10000000))
            list_condition_text.append(text)
            list_condition_start.append(start)
            list_condition_end.append(end)
    return list_condition_text, list_condition_start, list_condition_end

def split_audio(in_fname, out_fname, start, end):
    start = (start * 1000)
    end = (end * 1000)
    new_audio = AudioSegment.from_wav(in_fname)
    new_audio = new_audio[int(start) : int(end)]
    new_audio.export(out_fname, format= "wav")
    

def split_audio_old(in_fname, out_fname, start, end):
    with wave.open(in_fname, 'rb') as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        
        infile.setpos(int(start * framerate))

        data = infile.readframes(int((end - start) * framerate))
        
    # write the extracted data to a new file
    with wave.open(out_fname, 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)

def create_total_file(out_path, list_file_txt):
    print('STARTING EXPORT TO FILE ................')
    index = 0
    for element in tqdm(list_file_txt):
        if index == 0:
            with open(f"{out_path}/total_text_aligner.txt", 'w') as fta:
                for line in element:
                    fta.write(f'{line}\n')
                index = index + 1
        elif index == 1:
            with open(f"{out_path}/total_pitch_extractor.txt", 'w') as fpe:
                for line in element:
                    fpe.write(f"{line}\n")
                index = index + 1
        elif index == 2:
            with open(f"{out_path}/total_model_train.txt", 'w') as fmf:
                for line in element:
                    fmf.write(f"{line}\n")
                index = index + 1
    print('DONE ..........................')