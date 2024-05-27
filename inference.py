import denoiser
import os
from pydub import AudioSegment
from scipy.io import wavfile
import threading
import torch
import torchaudio
from denoiser.dsp import convert_audio
import json
from bin.component import *
from hifi_gan.vocoder import Generator
from attrdict import AttrDict
import phonemizer
from bin.models import load_ASR_models, load_F0_models, build_model
from munch import Munch
import yaml
import glob
from denoiser import pretrained


model_denoiser = pretrained.dns64().cpu()
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

# load StyleTTS
model_path = "Models/LibriTTS/epoch_2nd_00050.pth"
model_config_path = "Models/LibriTTS/config.yml"
device = input('Please enter your device (cpu/cuda): ')
if device == 'cpu':
    model_denoiser = pretrained.dns64().cpu()
elif device == 'cuda':
    model_denoiser = pretrained.dns64().cuda()

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




def get_dir(file_path):
    try:
        parent_dir, sub_dir = file_path.rsplit('/', 1)[0], file_path.rsplit('/', 1)[-1]
        file_name = sub_dir.split('.')[0]
        return parent_dir, file_name
    except:
        raise NotADirectoryError('file path have problem\nPlease check asset to get information or send your issue to my mail')

def convert_to_wav(parent_path, file_name):
    sound = AudioSegment.from_mp3(f'{parent_path}/{file_name}.mp3')
    sound.export(f"{parent_path}/{file_name}.wav", format = 'wav')
    return f"{parent_path}/{file_name}.wav"

h = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

cp_g = scan_checkpoint("src/LibriTTS", 'g_')

config_file = 'src/LibriTTS/config.json'
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device(device)
generator = Generator(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()

def denoiser_audio(name_file, WavFile):
    print('Denoising audio ....................')
    os.makedirs(f'denoiser_audio', exist_ok= True)
    wav, sr = torchaudio.load(WavFile)
    if device == 'cpu':
        wav = convert_audio(wav.cpu(), sr, model_denoiser.sample_rate, model_denoiser.chin)
    elif device == 'cuda':
        wav = convert_audio(wav.cuda(), sr, model_denoiser.sample_rate, model_denoiser.chin)

    with torch.no_grad():
        denoised = model_denoiser(wav[None])[0]
    # scipy.io.wavfile.write(f'{full_path_audio_new}/{name}', rate=sr, data= denoised)
    print('Saving audio ......................')
    torchaudio.save(f'denoiser_audio/{name_file}.wav', denoised.to('cpu'), model_denoiser.sample_rate)
    return f'denoiser_audio/{name_file}.wav'
    
if __name__ == '__main__':
    filePath = input('Please paste file path, you wanna inference: ')
    texttoSpeak = input('Please enter your text: ')
    
    print('Checking file audio input (You should convert it to wav file)')
    FullParentDir, filename = get_dir(file_path= filePath)
    if filePath.endswith('.wav'):
        
        print('Format file ---------- DONE')
        denoiser_current_audio =  denoiser_audio(name_file= filename,
                                                    WavFile = filePath)
        
    else:
        print('Warning: Your file is not right format')
        print('This program will auto repair this error')

        print(FullParentDir, filename)
        WavFile = convert_to_wav(parent_path= FullParentDir,
                                 file_name= filename)
        
        print("Progress convert file ---------- DONE")
        print(f'New file was save at {WavFile}. Please check it')
    

        denoiser_current_audio =  denoiser_audio(name_file= filename,
                                                    WavFile = WavFile)
    
    print(f"Denoiser file was save at {denoiser_current_audio}. You can check it ")
    print("Progress denoiser audio ---------- DONE")    
    print(f'Number character of text: {len(texttoSpeak)} ---------- DONE')
    print("Loading Hi-Fi GAN")
    
    cp_g = scan_checkpoint("src/LibriTTS", 'g_')

    config_file = 'src/LibriTTS/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    device = torch.device(device)
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    print('Loading Hi-Fi GAN ---------- DONE')
    print('Loading Model ---------- DONE')
    
    tokens = encode_text(text= texttoSpeak,
                     device= device,
                     global_phonemizer= global_phonemizer)
    reference_embeddings = encode_voice(denoiser_current_audio, model = model, device = device)

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
        print('Synthesizing audio %s ............' % key)
        current_path = str(os.getcwd()) + '/' + 'result'
        if os.path.exists(current_path):
            wavfile.write(f"{current_path}/result.wav", rate=24000, data= wave)
        else:
            os.makedirs(current_path)
            wavfile.write(f"{current_path}/result.wav", rate=24000, data= wave)

    print('Done synthenize completed')
    print(f'Result was saved at {current_path}/result.wav')