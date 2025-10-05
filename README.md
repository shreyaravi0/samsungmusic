# 🎶 AI-Powered Text, Image & Humming to Music Generator

 “Turn imagination into melody.”  
This AI-powered web app transforms **text descriptions**, **images**, or even your **humming** into expressive music compositions.  
You can fine-tune your output by choosing **genre**, **mood**, **tempo (BPM)**, and **duration**, resulting in truly personalized musical experiences.

---

## 🌟 Features

- 🎤 **Text-to-Music Generation** — Enter descriptive text to generate harmonized melodies.  
- 🖼️ **Image-to-Music Generation** — Upload an image, let the AI caption it, and convert it to music.  
- 🎙️ **Humming-to-Music Generation** — Upload a recorded hum, and it’ll create a full tune.  
- 🪄 **Mood Filters & Genre Effects** — Choose from multiple moods and genres to style your music.  
- ⏱️ **Tempo (BPM) Control** — Adjust the pace of your song to match its emotion.  
- ⏳ **Duration Limit** — Specify how long (in seconds) your song should be.  
- 🎧 **Playback & Speed Slider** — Listen, slow down, or speed up playback.  
- ⚙️ **FastAPI Backend** + **HTML/CSS/JS Frontend** — Easy to run locally.


---

## 🧰 Tech Stack

**Backend**
- 🐍 Python 3.10+
- ⚡ FastAPI
- 🤗 Hugging Face Transformers
- 🎵 Music21
- 🎧 Pydub
- 🎹 FluidSynth
- 🖼️ Pillow
- 🔊 Librosa

**Frontend**
- 🌐 HTML  
- 🎨 CSS  
- ⚡ JavaScript (Vanilla)

---

## 🧩 How It Works

1. **Input Phase**  
   - The user enters a text description, uploads an image, or provides an audio hum.  
   - You can specify:
     - **Genre** (cinematic, lofi, ambient, etc.)
     - **Mood** (calm, energetic, melancholic, etc.)
     - **Tempo (BPM)** to control playback speed  
     - **Duration (seconds)** to limit how long the generated music will be  

2. **Music Generation**  
   - Text is tokenized using a transformer (`sander-wood/text-to-music`) to generate **ABC notation**.  
   - For images, a **BLIP model** generates captions that describe the image.  
   - For humming, pitch frequencies are extracted using **Librosa**.

3. **Music Conversion**  
   - The ABC notation is parsed into a **MIDI sequence** using `music21`.  
   - A harmony layer is added automatically (triads and chords).  
   - MIDI is converted to **WAV audio** using `FluidSynth` and a `.sf2` soundfont.

4. **Mood & Genre Enhancements**  
   Music is post-processed using `pydub` filters:
   - 🎵 **Mood Filters**
     - `uplifting`: Slight volume boost (+6dB)
     - `melancholic`: Low-pass filter for soft tone
     - `energetic`: Higher gain (+10dB)
     - `calm`: Reduced volume (-4dB)
     - `mysterious`: High-pass filter for ethereal sound  
   - 🎧 **Genre Effects**
     - `cinematic`: Gradual fade in/out (film-like)
     - `lofi`: Muffled bass and softer volume
     - `electronic`: Bright and punchy output
     - `orchestral`: Smooth fade-in for realism
     - `ambient`: Long, smooth fade-in/out for spatial sound  

5. **Frontend Display**
   - Users can preview the song, adjust playback speed, and view musical notes.  
   - The generated **WAV file** can be downloaded for offline use.  

---

## ⚙️ Setup Instructions
1️⃣ Clone the repository
```bash
git clone https://github.com/shreyaravi0/samsungmusic.git
cd samsungmusic
```

2️⃣ Create a virtual environment
For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```
For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Download SoundFont
Download GeneralUser GS v1.472 → Download Link

Extract it, and set the path in app.py:
SOUNDFONT_PATH = ""

5️⃣ Run the backend server
```bash
uvicorn app:app --reload
```
Your FastAPI backend starts at:
👉 http://127.0.0.1:8000

6️⃣ Launch the frontend
Just open the file index.html in your browser 🌐

You can now:

📝 Enter text → Generate music
🖼️ Upload an image → Get AI-captioned melody
🎤 Upload humming audio → Convert it into structured music
🎧 Listen, adjust speed, view notes, and download the generated WAV file

🧪 Example Prompts
Try any of these creative prompts for beautiful results:

🎹 A soft piano tune for a rainy evening
🌆 A cyberpunk city at night with neon lights
🌅 A calm ocean sunrise melody
🎸 An energetic rock beat for motivation

Or upload an image of a forest, galaxy, or city —
the AI will caption it and generate a matching melody 🎵



Acknowledgements

🧠 Hugging Face Transformers

🎼 Music21 Toolkit

🎵 FluidSynth

⚡ FastAPI

🖼️ BLIP – Image Captioning Model


Team Members : (RV COLLEGE OF ENGINEERING)
Shreya Ravi
Aditya K 
Ansh Ravi Kashyap 
Poorvi Belur 




