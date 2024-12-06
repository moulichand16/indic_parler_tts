import os
import io
import nltk
import torch
import numpy as np
from typing import Iterator
from cog import BasePredictor, Input, Path
from pydub import AudioSegment
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor

# Download necessary NLTK data
nltk.download("punkt_tab")

class Predictor(BasePredictor):
    def setup(self):
        """Load the fine-tuned model and required components into memory."""
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
        # device="cuda:0"
        # torch_dtype=torch.bfloat16

        # Load the fine-tuned model
        finetuned_repo_id = "ai4bharat/indic-parler-tts"
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            finetuned_repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_repo_id)
        self.description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(finetuned_repo_id)

        self.sampling_rate = self.feature_extractor.sampling_rate

    def numpy_to_mp3(self, audio_array, sampling_rate):
        """Convert a NumPy array to MP3 bytes."""
        if np.issubdtype(audio_array.dtype, np.floating):
            max_val = np.max(np.abs(audio_array))
            audio_array = (audio_array / max_val) * 32767
            audio_array = audio_array.astype(np.int16)

        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="320k")
        mp3_bytes = mp3_io.getvalue()
        mp3_io.close()
        return mp3_bytes

    def generate_audio(self, text, description):
        """Generate audio for given text and description using the fine-tuned model."""
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

        inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        all_audio = []

        for chunk in chunks:
            prompt = self.tokenizer(chunk, return_tensors="pt").to(self.device)
            generation = self.model.generate(
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
        return self.numpy_to_mp3(combined_audio, sampling_rate=self.sampling_rate)

    def predict(
        self,
        text: str = Input(description="Input text to generate speech."),
        description: str = Input(description="Speaker description."),
    ) -> Iterator[Path]:
        """Run prediction."""
        mp3_bytes = self.generate_audio(text, description)

        # Save MP3 output to a file
        output_path = "/tmp/output.mp3"
        with open(output_path, "wb") as f:
            f.write(mp3_bytes)

        yield Path(output_path)
