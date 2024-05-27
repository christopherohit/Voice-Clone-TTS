from __future__ import unicode_literals
import yt_dlp
from tqdm import tqdm
from pytube import YouTube
import ffmpeg

ydl_opts = {
    'format': 'bestaudio/best',
#    'outtmpl': 'output.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}
def download_from_url(url, outPath):
    ydl.download([url])
    stream = ffmpeg.input(f'{outPath}.m4a')
    stream = ffmpeg.output(stream, f'{outPath}.wav')


with open('/home/nhan/voice-Research/viProcessing/example/subtitles.txt', 'r') as fp:
    list_data = fp.readlines()

list_speaker, list_link, list_gender = [], [], []

for line  in list_data:
    list_speaker.append(line.split('|')[0])
    list_link.append(line.split('|')[1])
    list_gender.append(line.split('|')[2])
    

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for index, file in tqdm(enumerate(list_link)):
        yt = YouTube(file)
        output_path = f'/vtca/nhanht/VTCA_viDataset/raw_wav/{list_speaker[index]}_{list_gender[index]}_{str(int(yt.length)).zfill(000000000)}'
        download_from_url(url= file, outPath= output_path)

