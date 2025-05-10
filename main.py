from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, torchaudio, logging, re, time

logging.getLogger("transformers").setLevel(logging.ERROR)

class NavaiSTT:
    
    def __init__(self, model_name="islomov/navaistt_v1_medium", target_sample_rate=16000, language="uz", isCapitalize=True):
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.language = language
        self.isCapitalize = isCapitalize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def preprocess_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            input_features = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                language=self.language,
            ).input_features.to(self.device)
            
            return input_features
            
        except Exception as e:
            raise
    
    def transcribe(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            chunk_duration = 30
            chunk_size = self.target_sample_rate * chunk_duration
            total_chunks = (waveform.size(0) + chunk_size - 1) // chunk_size

            transcriptions = []

            for i in range(total_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, waveform.size(0))
                chunk = waveform[start:end]

                input_features = self.processor(
                    chunk.numpy(),
                    sampling_rate=self.target_sample_rate,
                    return_tensors="pt",
                    language=self.language,
                ).input_features.to(self.device)

                with torch.no_grad():
                    predicted_ids = self.model.generate(input_features)

                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                if self.isCapitalize:
                    transcription = self.capitalize_sentences(transcription)

                transcriptions.append(transcription)

            return " ".join(transcriptions)

        except Exception as e:
            raise


    def capitalize_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        capitalized = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                capitalized.append(sentence[0].upper() + sentence[1:])

        return ' '.join(capitalized)


if __name__ == "__main__":
    starting_time = time.time()
    audio_file = "audio.wav"

    transcriber = NavaiSTT()
    transcription = transcriber.transcribe(audio_file)

    print(f"Transcription: {transcription}")
    print(f"Time taken: {time.time() - starting_time:.2f} seconds")