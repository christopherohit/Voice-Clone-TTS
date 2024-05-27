import librosa
from api.eng import preprocessing_text
from scipy.io import wavfile
import shutil
import json
from pathlib import Path
from fastapi.responses import FileResponse
from bin.models import *
import soundfile as sf
from attrdict import AttrDict
import torchaudio
from hifi_gan.vocoder import Generator
import torch

import os

h = None

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

config_file = 'src/LibriTTS/config.json'
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device('cpu')
generator = Generator(h).to(device)

state_dict_g = load_checkpoint("src/LibriTTS/g_00935000", device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()

model_path = "Models/LibriTTS/epoch_2nd_00050.pth"
model_config_path = "Models/LibriTTS/config.yml"

config = yaml.safe_load(open(model_config_path))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

params = torch.load(model_path, map_location='cpu')
params = params['net']
for key in model:
    if key in params:
        if not "discriminator" in key:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key])
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)
        try:
            with torch.no_grad():
                ref = model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)
        except:
            continue
    return reference_embeddings


async def generate_2(text, file):
    
    with open('requestFile/response.wav', "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    path = 'result/'
    inputText = str(text)
    audio = 'requestFile/response.wav'
    tokens = preprocessing_text(text_input= inputText)
    audict = {}
    name = audio.split('/')[-1].replace('.wav', '')
    audict[name] = audio
    reference_embeddings = compute_style(audict)    
    
    converted_samples = {}
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        m = length_to_mask(input_lengths).to(device)
        t_en = model.text_encoder(tokens, input_lengths, m)
            
        for key, (ref, _) in reference_embeddings.items():
            
            s = ref.squeeze(1)
            style = s
            
            d = model.predictor.text_encoder(t_en, style, input_lengths, m)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            style = s.expand(en.shape[0], en.shape[1], -1)

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))


            c = out.squeeze()
            y_g_hat = generator(c.unsqueeze(0))
            y_out = y_g_hat.squeeze().cpu().numpy()

            c = out.squeeze()
            y_g_hat = generator(c.unsqueeze(0))
            y_out = y_g_hat.squeeze()
            
            converted_samples[key] = y_out.cpu().numpy()
    
    for key, wave in converted_samples.items():
        try:
            wavfile.write(f'{path}/result.wav', 24000, wave)
            output_audio = f'{path}result.wav'
            # output_base64 = base64.b64encode(output_audio.read_bytes())
            return FileResponse(output_audio, filename='result/result.wav', media_type="audio/wav")
        except:
            raise Exception()
