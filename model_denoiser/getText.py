import os
import azure.cognitiveservices.speech as speechsdk
import json
import time
import subprocess
import sys




os.environ['SPEECH_KEY'] = "34cc0a5151f74011911abde9e6bde050" # visible in this process + all children
os.environ['SPEECH_REGION'] = "eastasia"


def stt(input_file, out_file):
    """
    The `stt` function is a Python code that performs speech-to-text (STT) conversion using Azure
    Cognitive Services.
    
    :param input_file: The `input_file` parameter is the path to the audio file that you want to perform
    speech-to-text (STT) on. It should be a string representing the file path
    :param out_file: The `out_file` parameter is the path and filename of the output file where the
    recognized speech will be saved
    """
    
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.request_word_level_timestamps()
    speech_config.speech_recognition_language="vi-VN"

    audio_config = speechsdk.audio.AudioConfig(filename= input_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    # speech_recognizer.recognized.connect(lambda evt: print('JSON: {}'.format(evt.result.json)))
    done = False

    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        global done
        done = True

    print('Starting ..............')
    # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    with open (out_file, 'w') as fp:
        speech_recognizer.recognizing.connect(lambda x: print(end=" "))
        speech_recognizer.recognized.connect(lambda evt: fp.write(f'{evt.result.text}|{evt.result.offset}|{evt.result.duration}\n'))
        speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
        
        
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        speech_recognizer.start_continuous_recognition()
        while not done:
            time.sleep(.5)