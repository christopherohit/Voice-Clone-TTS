import collections
import yaml
from munch import Munch
import torch
from torch import nn
import torch.nn.functional as F
import IPython.display as ipd
import torchaudio
import sys
sys.path.append('api')
import text
import librosa

from bin.models import *
from bin.utils import *
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True, language_switch="remove-flags", words_mismatch='ignore')

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

device = 'cpu'

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(char)
        return indexes

textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_dicts, model):
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

def preprocessing_text(text_input):
    
    cleaned_text = text._clean_text(text_input, cleaner_names=['english_cleaners2'])
    tokens = textclenaer(cleaned_text)
    tokens.insert(0, 0)
    tokens.append(0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    return tokens

def preprocessing_audio(audio, model):
    
    audict = {}
    name = audio.split('/')[-1].replace('.wav', '')
    audict[name] = audio
    reference_embeddings = compute_style(audict, model= model)
    return reference_embeddings

def Cloning(model, tokens, reference_embeddings ,generator):
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
            return converted_samples

def display_it(converted_samples):
    for key, wave in converted_samples.items():
        return wave