import yaml
from bin.models import load_ASR_models, load_F0_models, build_model
import torch
from munch import Munch
import phonemizer
from bin.component import *
import os
from attrdict import AttrDict
import glob
import json
from hifi_gan.vocoder import Generator
from scipy.io import wavfile


global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

# load StyleTTS
model_path = "Models/LibriTTS/epoch_2nd_00050.pth"
model_config_path = "Models/LibriTTS/config.yml"
device = 'cpu'

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




text_to_synthenize = input('Please enter text to synthenize: ')
path_to_wave = input('Path of audio: ')

tokens = encode_text(text= text_to_synthenize,
                     device= device,
                     global_phonemizer= global_phonemizer)

reference_embeddings = encode_voice(path_to_wave, model = model, device = device)

#Process

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