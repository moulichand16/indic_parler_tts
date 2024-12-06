from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import Thread
from queue import Queue
import os
import uuid
import torch
import numpy as np
import nltk
from pydub import AudioSegment
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor

nltk.download("punkt_tab")

app = FastAPI(title="Text-to-Speech API")

# Model Initialization
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

finetuned_repo_id = "ai4bharat/indic-parler-tts"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    finetuned_repo_id, attn_implementation="eager", torch_dtype=torch_dtype
).to(device)
tokenizer = AutoTokenizer.from_pretrained(finetuned_repo_id)
description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
feature_extractor = AutoFeatureExtractor.from_pretrained(finetuned_repo_id)

sampling_rate = feature_extractor.sampling_rate

# Request Model
class TTSRequest(BaseModel):
    text: str
    description: str

# Queue for managing requests
request_queue = Queue()
results = {}

def numpy_to_mp3(audio_array, sampling_rate):
    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        audio_array = (audio_array / max_val) * 32767
        audio_array = audio_array.astype(np.int16)

    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1,
    )
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")
    return mp3_io.getvalue()

def process_tts_request(request_id, text, description):
    try:
        chunk_size = 25
        sentences = nltk.sent_tokenize(text)
        chunks = []
        curr_sentence = ""

        for sentence in sentences:
            candidate = " ".join([curr_sentence, sentence])
            if len(candidate.split()) >= chunk_size:
                chunks.append(curr_sentence)
                curr_sentence = sentence
            else:
                curr_sentence = candidate

        if curr_sentence:
            chunks.append(curr_sentence)

        inputs = description_tokenizer(description, return_tensors="pt").to(device)
        all_audio = []

        for chunk in chunks:
            prompt = tokenizer(chunk, return_tensors="pt").to(device)
            generation = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                do_sample=True,
                return_dict_in_generate=True,
            )
            if hasattr(generation, "sequences") and hasattr(generation, "audios_length"):
                audio = generation.sequences[0, :generation.audios_length[0]]
                audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
                all_audio.append(audio_np.flatten() if len(audio_np.shape) > 1 else audio_np)

        combined_audio = np.concatenate(all_audio)
        mp3_bytes = numpy_to_mp3(combined_audio, sampling_rate=sampling_rate)

        # Save the result
        results[request_id] = mp3_bytes
    except Exception as e:
        results[request_id] = str(e)

def worker():
    while True:
        request_id, text, description = request_queue.get()
        process_tts_request(request_id, text, description)
        request_queue.task_done()

# Start worker thread
thread = Thread(target=worker, daemon=True)
thread.start()

@app.post("/generate-audio")
def generate_audio(request: TTSRequest):
    request_id = str(uuid.uuid4())
    request_queue.put((request_id, request.text, request.description))
    return {"request_id": request_id, "message": "Request received. Check status using /status/{request_id}."}

@app.get("/status/{request_id}")
def get_status(request_id: str):
    if request_id in results:
        if isinstance(results[request_id], bytes):
            file_path = f"/tmp/{request_id}.mp3"
            with open(file_path, "wb") as f:
                f.write(results[request_id])
            return {"status": "completed", "file_path": file_path}
        else:
            return {"status": "failed", "error": results[request_id]}
    else:
        return {"status": "pending", "message": "Request is still being processed."}
