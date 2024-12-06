import io
import os
import math
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
import spaces
import gradio as gr
import torch
import nltk
import certifi

from parler_tts import ParlerTTSForConditionalGeneration
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed

nltk.download('punkt_tab')

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

repo_id = "ai4bharat/indic-parler-tts-pretrained"
finetuned_repo_id = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
).to(device)
finetuned_model = ParlerTTSForConditionalGeneration.from_pretrained(
    finetuned_repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42

default_text = "Please surprise me and speak in whatever voice you enjoy."
examples = [
    [
        "मुले बागेत खेळत आहेत आणि पक्षी किलबिलाट करत आहेत.",
        "Sunita speaks slowly in a calm, moderate-pitched voice, delivering the news with a neutral tone. The recording is very high quality with no background noise.",
        3.0
    ],
]


finetuned_examples = [
    [
        "मुले बागेत खेळत आहेत आणि पक्षी किलबिलाट करत आहेत.",
        "Sunita speaks slowly in a calm, moderate-pitched voice, delivering the news with a neutral tone. The recording is very high quality with no background noise.",
        3.0
    ],

]


def numpy_to_mp3(audio_array, sampling_rate):
    # Normalize audio_array if it's floating-point
    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        audio_array = (audio_array / max_val) * 32767  # Normalize to 16-bit range
        audio_array = audio_array.astype(np.int16)

    # Create an audio segment from the numpy array
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )

    # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")

    # Get the MP3 bytes
    mp3_bytes = mp3_io.getvalue()
    mp3_io.close()

    return mp3_bytes

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

@spaces.GPU
def generate_base(text, description,):
    # Initialize variables
    chunk_size = 25  # Process max 25 words or a sentence at a time
    
    # Tokenize the full text and description
    inputs = description_tokenizer(description, return_tensors="pt").to(device)

    sentences_text = nltk.sent_tokenize(text) # this gives us a list of sentences
    curr_sentence = ""
    chunks = []
    for sentence in sentences_text:
        candidate = " ".join([curr_sentence, sentence])
        if len(candidate.split()) >= chunk_size:
            chunks.append(curr_sentence)
            curr_sentence = sentence
        else:
            curr_sentence = candidate

    if curr_sentence != "":
        chunks.append(curr_sentence)
        
    print(chunks)

    all_audio = []
    
    # Process each chunk
    for chunk in chunks:
        # Tokenize the chunk
        prompt = tokenizer(chunk, return_tensors="pt").to(device)
        
        # Generate audio for the chunk
        generation = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            return_dict_in_generate=True
        )
            
        # Extract audio from generation
        if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
            audio = generation.sequences[0, :generation.audios_length[0]]
            audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            all_audio.append(audio_np)
    
    # Combine all audio chunks
    combined_audio = np.concatenate(all_audio)
    
    # Convert to expected format and yield
    print(f"Sample of length: {round(combined_audio.shape[0] / sampling_rate, 2)} seconds")
    yield numpy_to_mp3(combined_audio, sampling_rate=sampling_rate)



def generate_finetuned(text, description):
    # Initialize variables
    chunk_size = 25  # Process max 25 words or a sentence at a time
    
    # Tokenize the full text and description
    inputs = description_tokenizer(description, return_tensors="pt").to(device)

    sentences_text = nltk.sent_tokenize(text) # this gives us a list of sentences
    curr_sentence = ""
    chunks = []
    for sentence in sentences_text:
        candidate = " ".join([curr_sentence, sentence])
        if len(candidate.split()) >= chunk_size:
            chunks.append(curr_sentence)
            curr_sentence = sentence
        else:
            curr_sentence = candidate

    if curr_sentence != "":
        chunks.append(curr_sentence)
        
    print(chunks)
    
    all_audio = []
    
    # Process each chunk
    for chunk in chunks:
        # Tokenize the chunk
        prompt = tokenizer(chunk, return_tensors="pt").to(device)

        # Generate audio for the chunk
        generation = finetuned_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            return_dict_in_generate=True
        )

        # Extract audio from generation
        if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
            audio = generation.sequences[0, :generation.audios_length[0]]
            audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            all_audio.append(audio_np)
    
    # Combine all audio chunks
    combined_audio = np.concatenate(all_audio)
    
    # Convert to expected format and yield
    print(f"Sample of length: {round(combined_audio.shape[0] / sampling_rate, 2)} seconds")
    yield numpy_to_mp3(combined_audio, sampling_rate=sampling_rate)


if __name__ == "__main__":
    # Example text and description
    text = "Hello, this is a test."
    description = "Calm voice"

    # Call the generate_finetuned function
    audio_generator = generate_finetuned(text, description)

    # Process the generated audio
    for audio in audio_generator:
        # Here you can save the audio or process it further
        with open("./output.mp3", "wb") as f:
            f.write(audio)