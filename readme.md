# Uzbek Speech-to-Text (STT) Project - Navaistt

This project provides Speech-to-Text (STT) functionality for Uzbek language audio files. It utilizes the `islomov/navaistt_v1_medium` model from Hugging Face.

## Description

The core STT model ([`islomov/navaistt_v1_medium`](https://huggingface.co/islomov/navaistt_v1_medium)) is optimized for processing audio segments up to 30 seconds long. This project extends its capability to transcribe longer audio files by:
1.  Splitting the input audio into 30-second chunks.
2.  Processing each chunk individually using the STT model.
3.  Combining the transcribed text from all chunks to produce the final transcription.

## Model Used

The STT functionality is powered by the [islomov/navaistt_v1_medium](https://huggingface.co/islomov/navaistt_v1_medium) model available on Hugging Face.

## Features

-   Transcription of Uzbek language audio.
-   Handles audio files longer than 30 seconds through automatic chunking and result aggregation.

## Requirements

-   Python 3.x
-   Libraries specified in `requirements.txt` (if available). Key libraries likely include:
    -   `transformers`
    -   `torch`
    -   `torchaudio`

## Usage

1.  **Clone the repository (if applicable) or download the `main.py` file.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    # Or install libraries manually, e.g., pip install transformers torch pydub
    ```
3. **Code Usage**
    ```python
    if __name__ == "__main__":
        starting_time = time.time()
        audio_file = "audio.wav"

        transcriber = NavaiSTT()
        transcription = transcriber.transcribe(audio_file)

        print(f"Transcription: {transcription}")
        print(f"Time taken: {time.time() - starting_time:.2f} seconds")
    ```
4.  **Run the script:**
    ```bash
    python main.py
    ```

## For more information, go to the official [website](https://uz-speech.web.app/navaistt01m).