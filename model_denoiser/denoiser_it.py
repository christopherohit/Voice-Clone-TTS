import torch
import argparse
import torchaudio
import glob
from denoiser import pretrained
import os
from tqdm import tqdm
from denoiser.dsp import convert_audio


model = pretrained.dns64().cuda()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default= '/vtca/nhanht/VIVO/wave/')
    parser.add_argument('--makeCopy', default= True, type= bool),
    parser.add_argument('--extendsion', default='wav'),
    parser.add_argument('--existID', default=True, type= bool)
    
    args = parser.parse_args()
    
    Flag = True
    if args.existID:
        file_in_path = glob.glob(f'{args.path}/*/*.{args.extendsion}')
    else:
        file_in_path = glob.glob(f'{args.path}/*.{args.extendsion}')
    
    
    if args.makeCopy: 
        repars = args.path.split('/')[:-1]
        
        newPath = '/'.join(repars)
        os.makedirs(f'{newPath}/audio_new', exist_ok= True)
        full_path_audio = f'{newPath}/audio_new'
        for audio in tqdm(file_in_path):
            id_speaker = audio.split('/')[-2]
            name = audio.split('/')[-1]
            wav, sr = torchaudio.load(audio)
            os.makedirs(f'{full_path_audio}/{id_speaker}', exist_ok= True)
            full_path_audio_new = f'{full_path_audio}/{id_speaker}'
            wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
            with torch.no_grad():
                denoised = model(wav[None])[0]
            # scipy.io.wavfile.write(f'{full_path_audio_new}/{name}', rate=sr, data= denoised)
            torchaudio.save(f'{full_path_audio_new}/{name}', denoised.to('cpu'), model.sample_rate)
    else:
        for audio in tqdm(file_in_path):
            wav, sr = torchaudio.load(audio)
            wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
            with torch.no_grad():
                denoised = model(wav[None])[0]
            torchaudio.save(audio, denoised.to('cpu'), model.sample_rate)
            