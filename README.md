# Hollis

**Hollis** is an acoustic cognitive architecture designed to transform raw environmental audio into semantic understanding. Built on the **Newbound/Flowlang** stack, Hollis uses binaural audio processing to track entities, analyze environmental context, and perform local speech transcription.

Unlike standard voice assistants that simply wait for a wake word, Hollis maintains a persistent "World State," tracking the presence of specific acoustic entities (people), the "vibe" of the room (atmosphere), and the flow of discourse over time.

## 🚀 Features

* **Binaural Perception:** Fuses audio from two microphones to estimate direction of arrival (DOA) and separate sound sources.
* **Acoustic Entity Tracking:** Identifies and tracks unique sound sources ("Entities") using spectral fingerprinting and spatial logic.
* **Context Awareness:** Generates high-level briefings on the environment (e.g., "The room is library-quiet" vs "Active conversation").
* **Local Transcription:** Uses **OpenAI Whisper** (via `whisper-rs`) for privacy-focused, offline speech-to-text.
* **Cognitive Loop:** Implements a "Pulse" architecture (Ingest → Synthesize → Emit) to process sensory data into upstream insights.

## 🛠️ Installation

### 1. Prerequisites
* **Rust:** Ensure you have the latest stable Rust and Cargo installed (`rustup update`).
* **System Dependencies:** You may need `alsa-lib` or `libasound2-dev` (Linux) for the `cpal` audio backend.
* **Microphones:** Hollis works best with a stereo pair of microphones to enable spatial tracking, though it can function with a single input.

### 2. Quick Start
This repository includes a bootstrap script to set up the Newbound environment and build the project in one step.

1.  Clone this repository:
    ```bash
    git clone https://github.com/mraiser/hollis.git
    cd hollis
    ```

2.  Run the installer:
    ```bash
    ./install.sh
    ```
    *This script will fetch the necessary Newbound core dependencies, compile the project, and launch the application.*

## 🧠 Model Setup (If not using the install script)

Hollis runs the speech recognition model locally. **You must either install with the install.sh script or download the model file manually** before the transcription features will work.

1.  **Download the Model:**
    Get the `ggml-medium.en.bin` model (compatible with `whisper.cpp`).
    * *Recommended Source:* [Hugging Face - ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

2.  **Place the File:**
    By default, Hollis looks for the model at `models/ggml-medium.en.bin` relative to the runtime directory.

3.  **Configure the Path:**
    If you place the model elsewhere, or wish to use a different size (e.g., `base.en`, `large-v3`), edit the configuration file located at:
    `runtime/hollis/botd.properties`

    Add or update the following line:
    ```properties
    model=/absolute/path/to/your/ggml-model.bin
    ```

## ⚙️ Configuration

Configuration is handled in `runtime/hollis/botd.properties`. The system will generate a default file if one is missing, but you can tune the following parameters:

| Property | Description | Default |
| :--- | :--- | :--- |
| `model` | Path to the GGML Whisper model binary. | `models/ggml-medium.en.bin` |
| `mic1` | Name (partial match) of the primary/left microphone. | `default` |
| `mic2` | Name (partial match) of the secondary/right microphone. | *(None)* |
| `mic_distance` | Distance between microphones in meters (for spatial math). | `0.6` |
| `memory_file` | Filename for the JSON entity database. | `hollis_memory.json` |

### 🧠 LLM Configuration

Hollis supports multiple LLM backends for its cognitive functions. You can configure the provider using the `LLM` property in `botd.properties`.

| Property | Description | Default |
| --- | --- | --- |
| `LLM` | The LLM backend provider. Options: `GEMINI`, `OLLAMA`, or `CUSTOM`. | `GEMINI` |

#### Provider Specific Settings

**1. Google Gemini (Default)**
Required if `LLM=GEMINI`.

* `GEMINI_API_KEY`: Your Google AI Studio API key.
* *(Note: Currently uses `gemini-2.5-flash`)*

**2. Ollama (Local)**
Required if `LLM=OLLAMA`.

* `OLLAMA_URL`: The full API endpoint (e.g., `http://localhost:11434/api/generate`).
* `OLLAMA_MODEL`: The model tag to use (e.g., `llama3`, `mistral`).

**3. Custom / External**
Used if `LLM` is set to anything else.

* `LLM_CTL`: Route to a custom Flowlang command in the format `lib:ctl:cmd`.
* *Example:* `my_lib:my_controller:my_llm_wrapper`

## 🏗️ Architecture

Hollis is composed of three primary cognitive layers:

1.  **The Sensor (Ear):**
    Captures raw audio frames, performs FFT analysis, calculates RMS/Loudness, and handles hardware I/O via `cpal`.

2.  **The Perception Engine (stem):**
    * **Fusion:** Combines stereo inputs into a single "Super Frame."
    * **VAD:** Detects voice activity and meaningful transient sounds (claps, knocks).
    * **Briefing:** Summarizes the "Atmosphere" (noise floor, chaos level).

3.  **The Cortex (Brain):**
    * **Entity Resolution:** Uses cosine similarity on spectral fingerprints to identify if the person speaking is a known "Entity" or a new guest.
    * **Discourse Analysis:** Aggregates transcripts to summarize the topic of conversation.
    * **Situation Room:** Maintains a high-level key-value store of the current reality (Occupancy, Atmosphere, Discourse).

## 📄 License

MIT License

Copyright (c) 2026 Marc Raiser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
