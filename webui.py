# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Pre-trained Voice', '3s Fast Reproduction', 'Cross-lingual Reproduction', 'Natural Language Control']
instruct_dict = {'Pre-trained Voice': '1. Select a pre-trained voice\n2. Click the Generate Audio button',
                 '3s Fast Reproduction': '1. Select a prompt audio file, or record a prompt audio (no longer than 30s). If both are provided, the prompt audio file will be prioritized\n2. Enter the prompt text\n3. Click the Generate Audio button',
                 'Cross-lingual Reproduction': '1. Select a prompt audio file, or record a prompt audio (no longer than 30s). If both are provided, the prompt audio file will be prioritized\n2. Click the Generate Audio button',
                 'Natural Language Control': '1. Select a pre-trained voice\n2. Enter the instruction text\n3. Click the Generate Audio button'}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['Natural Language Control']:
        if instruct_text == '':
            gr.Warning('You are using the Natural Language Control mode, please enter instruction text')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using the Natural Language Control mode, the prompt audio/prompt text will be ignored')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['Cross-lingual Reproduction']:
        if instruct_text != '':
            gr.Info('You are using the Cross-lingual Reproduction mode, the instruction text will be ignored')
        if prompt_wav is None:
            gr.Warning('You are using the Cross-lingual Reproduction mode, please provide a prompt audio')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('You are using the Cross-lingual Reproduction mode, please ensure that the synthesized text and prompt text are in different languages')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s Fast Reproduction', 'Cross-lingual Reproduction']:
        if prompt_wav is None:
            gr.Warning('The prompt audio is empty, did you forget to input the prompt audio?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('The prompt audio sample rate {} is lower than {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['Pre-trained Voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using the Pre-trained Voice mode, the prompt text/prompt audio/instruction text will be ignored!')
        if sft_dropdown == '':
            gr.Warning('No pre-trained tones are available!')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s Fast Reproduction']:
        if prompt_text == '':
            gr.Warning('The prompt text is empty, did you forget to input the prompt text?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using the 3s Fast Reproduction mode, the pre-trained voice/instruction text will be ignored!')

    if mode_checkbox_group == 'Pre-trained Voice':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s Fast Reproduction':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-lingual Reproduction':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("#### Please enter the text to be synthesized, select the inference mode, and follow the steps provided")

        tts_text = gr.Textbox(label="Enter Text to Synthesize", lines=1, value="I am a newly released generative speech model by the Tongyi Speech Team, providing a comfortable and natural speech synthesis capability.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select Inference Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select Pre-trained Voice', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Is it streaming reasoning?', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed ​​adjustment (only supports non-streaming inference)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random Inference Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload Prompt Audio File (Sampling rate no less than 16kHz)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record Prompt Audio File')
        prompt_text = gr.Textbox(label="Enter Prompt Text", lines=1, placeholder="Please enter prompt text that matches the content of the prompt audio, automatic recognition is currently not supported...", value='')
        instruct_text = gr.Textbox(label="Enter Instruct Text", lines=1, placeholder="Please enter instruct text.", value='')

        generate_button = gr.Button("Generate Audio")

        audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
