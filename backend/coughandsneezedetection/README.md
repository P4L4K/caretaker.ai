# SonicGuard — Real-Time Cough & Sneeze Detection

A real-time audio classification system that uses a deep learning CRNN model to classify live microphone audio into **Cough**, **Sneeze**, **Normal Talking**, and **Background Noise**. Features a premium dark-themed dashboard with live waveform/spectrogram visualization and an instant alert sidebar.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Using the Dashboard](#using-the-dashboard)
- [Training on Your Own Data](#training-on-your-own-data)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Description |
|:---|:---|
| **CRNN Model** | CNN (SeparableConv2D + Residual Blocks) + Bidirectional GRU for spectral-temporal classification |
| **Log-Mel Spectrograms** | 25ms window, 10ms hop, 64 mel bands — optimised for capturing transient cough/sneeze spikes |
| **Real-Time Streaming** | 500ms analysis windows with 50% overlap via WebSocket (Socket.IO) |
| **Confidence Thresholding** | Alerts only fire when Cough/Sneeze confidence ≥ 85%, preventing false alarms |
| **Live Visualiser** | Canvas-based waveform + frequency spectrogram, updated in real time |
| **Alert Sidebar** | Timestamped, color-coded log entries (Red = Cough, Amber = Sneeze) |
| **Detection Stats** | Live counters for Coughs, Sneezes, and Total Alerts |
| **Reset** | One-click reset clears all UI state and server buffers |
| **Data Augmentation** | White noise, time stretch, pitch shift, volume perturbation, background mixing |

---

## Project Structure

```
cough and sneeze detection/
│
├── app.py                              # Flask-SocketIO backend server
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── model/
│   ├── __init__.py                     # Package exports
│   ├── architecture.py                 # CRNN model definition
│   ├── preprocessing.py               # Audio → Log-Mel Spectrogram pipeline
│   └── weights.weights.h5             # Model weights (generated)
│
├── training/
│   ├── __init__.py
│   ├── augment.py                      # Data augmentation utilities
│   ├── train.py                        # Full training script
│   └── generate_demo_weights.py        # Generate initial weights
│
├── static/
│   ├── css/
│   │   └── style.css                   # Premium dark-theme styles
│   └── js/
│       ├── main.js                     # Frontend logic & WebSocket client
│       └── audio-processor.js          # AudioWorklet background processor
│
└── templates/
    └── index.html                      # Dashboard HTML
```

---

## Prerequisites

- **Python 3.9+** — [Download Python](https://www.python.org/downloads/)
- **pip** — comes with Python
- **A modern web browser** — Chrome, Edge, or Firefox (for microphone access)
- **A working microphone** — built-in or external

> **Note:** The browser will ask for microphone permission. On `localhost`, HTTPS is not required.

---

## Setup & Installation

### 1. Clone or download the project

```bash
cd "cough and sneeze detection"
```

### 2. (Recommended) Create a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `flask`, `flask-socketio`, `flask-cors`, `eventlet` — web server & WebSocket
- `numpy`, `scipy`, `librosa`, `soundfile` — audio processing
- `tensorflow` — deep learning model

### 4. Generate model weights

On the **first run only**, generate the initial model weights:

```bash
python training/generate_demo_weights.py
```

This creates `model/weights.weights.h5`. The model starts with random weights and will classify randomly until trained on real data (see [Training](#training-on-your-own-data)).

---

## Running the Application

```bash
python app.py
```

You will see:

```
====================================================
  SonicGuard CRNN -- Loading ...
====================================================
  [OK] Loaded weights from model\weights.weights.h5
  [OK] Model warmed up -- graph traced

  >> Server starting at http://localhost:5001
     Chunk size: 500ms  |  Overlap: 50%
```

**Open your browser at** → [http://localhost:5001](http://localhost:5001)

---

## Using the Dashboard

### Starting Monitoring
1. Click **Start Monitoring** (green button in the header)
2. Grant microphone permission when the browser prompts
3. The **waveform** and **spectrogram** canvases will animate with your live audio

### Understanding the Display

| Section | What it shows |
|:---|:---|
| **Waveform** | Real-time audio signal from your microphone |
| **Spectrogram** | Frequency content over time (blue = low energy, red = high energy) |
| **Classification Result** | Current predicted class with an icon and confidence % |
| **Detection Stats** | Running counters for Coughs, Sneezes, and Total Alerts |
| **Confidence Bars** | Live confidence % for all four classes (Cough, Sneeze, Talking, Noise) |
| **Alert Sidebar** | Timestamped log of all Cough/Sneeze detections above 85% confidence |

### Alert Color Coding
- **Red tag** → Cough detected
- **Amber/Yellow tag** → Sneeze detected

### Controls
| Button | Action |
|:---|:---|
| **Start Monitoring** | Begin capturing and classifying audio |
| **Stop** | Pause monitoring (keeps logs) |
| **Reset** | Stop monitoring AND clear all data (logs, counters, visualisers, server buffer) |
| **Clear All** (sidebar) | Clear only the alert log entries |

---

## Training on Your Own Data

The demo weights produce random predictions. To get accurate classification, train on real audio:

### 1. Organise your audio dataset

```
data/
├── cough/          # .wav, .mp3, or .flac files of cough sounds
├── sneeze/         # sneeze sounds
├── talking/        # normal speech
└── noise/          # background noise (fans, typing, street, etc.)
```

**Recommended datasets:**
- [COUGHVID](https://zenodo.org/record/4048312) — crowd-sourced cough recordings
- [ESC-50](https://github.com/karolpiczak/ESC-50) — environmental sounds
- [AudioSet](https://research.google.com/audioset/) — large-scale labelled audio
- [FSDKaggle2019](https://zenodo.org/record/3612637) — Freesound audio clips

### 2. Run the training script

```bash
python training/train.py --data_dir data --epochs 50 --batch_size 32
```

**Available options:**

| Flag | Default | Description |
|:---|:---|:---|
| `--data_dir` | `data` | Path to dataset root |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--augment_factor` | `5` | Augmented copies per sample |
| `--output` | `model/weights.weights.h5` | Where to save trained weights |

### 3. Restart the server

After training completes, restart `python app.py` to load the new weights.

---

## Configuration

Key settings in `app.py` you can adjust:

```python
CONFIDENCE_THRESHOLD = 0.85      # Min confidence to trigger an alert (0.0–1.0)
CHUNK_DURATION       = 0.5       # Analysis window in seconds
OVERLAP_RATIO        = 0.5       # Overlap between windows (0.0–1.0)
```

Key settings in `model/preprocessing.py`:

```python
SAMPLE_RATE    = 16_000   # Audio sample rate
WIN_LENGTH     = 400      # FFT window (25ms)
HOP_LENGTH     = 160      # FFT hop (10ms)
N_MELS         = 64       # Mel frequency bands
```

---

## Troubleshooting

| Issue | Solution |
|:---|:---|
| **Port 5001 already in use** | Change the port in `app.py` (line with `socketio.run(...)`) |
| **No microphone access** | Ensure you're on `localhost` (not an IP) and allow the browser permission |
| **Predictions seem random** | Expected with demo weights — train on real data first |
| **TensorFlow warnings in console** | Normal — TF prints hardware optimisation suggestions to stderr |
| **"No weights found" at startup** | Run `python training/generate_demo_weights.py` first |
| **Slow first classification** | The first inference compiles the TF graph — subsequent ones are fast |

---

## License

This project is for educational and research purposes.
