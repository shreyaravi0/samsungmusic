from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlipProcessor, BlipForConditionalGeneration
from music21 import converter, midi, stream, note, chord
from midi2audio import FluidSynth
from pydub import AudioSegment
import re
import os
import librosa

# ----------------------------
# Setup directories
# ----------------------------
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(title="AI Music Generator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (WAV/MP3)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ----------------------------
# Load Models
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained('sander-wood/text-to-music')
model = AutoModelForSeq2SeqLM.from_pretrained('sander-wood/text-to-music')

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# FluidSynth SoundFont
SOUNDFONT_PATH = "GeneralUser GS v1.472.sf2"
if not os.path.exists(SOUNDFONT_PATH):
    raise FileNotFoundError(f"SoundFont not found at {SOUNDFONT_PATH}")
fs = FluidSynth(SOUNDFONT_PATH)

# ----------------------------
# Helper Functions
# ----------------------------
def generate_abc_from_text(text_input: str):
    """Generate ABC notation and notes from text using text-to-music model"""
    input_ids = tokenizer(text_input, return_tensors='pt', truncation=True, max_length=512)['input_ids']
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
    eos_token_id = model.config.eos_token_id

    for _ in range(256):
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        probs = torch.nn.Softmax(dim=-1)(outputs.logits[0, -1])
        sampled_id = int(torch.argmax(probs))
        decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[sampled_id]])), 1)
        if sampled_id == eos_token_id:
            break

    tune_body = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    tune_body = re.sub(r'[^A-Ga-gz,\s|]', '', tune_body)
    notes = tune_body.split()
    abc_content = f"X:1\nL:1/8\nM:4/4\nK:C\n{tune_body}"
    return abc_content, notes


def abc_to_wav_mp3(abc_content: str, output_name: str,
                   genre: str, mood: str, tempo: int, duration: int):
    """Convert ABC → MIDI → WAV → MP3 with playback speed, genre, mood, tempo, duration effects"""
    score = converter.parse(abc_content, format='abc')

    # Apply tempo (default ~90 BPM)
    if tempo > 0:
        factor = max(0.5, tempo / 90)
        for el in score.recurse().notes:
            el.quarterLength /= factor

    # Simple harmonization
    melody = score.flatten().notes
    harmony_stream = stream.Part()
    for m_note in melody:
        if isinstance(m_note, note.Note):
            triad = chord.Chord([m_note.pitch, m_note.pitch.transpose(4), m_note.pitch.transpose(7)])
            triad.quarterLength = m_note.quarterLength
            harmony_stream.append(triad)

    melody_part = stream.Part()
    for n in melody:
        melody_part.append(n)

    final_score = stream.Score()
    final_score.insert(0, melody_part)
    final_score.insert(0, harmony_stream)

    # Save MIDI
    midi_file = os.path.join(STATIC_DIR, f"{output_name}.mid")
    mf = midi.translate.streamToMidiFile(final_score)
    mf.open(midi_file, 'wb')
    mf.write()
    mf.close()

    # Convert MIDI → WAV
    wav_file = os.path.join(STATIC_DIR, f"{output_name}.wav")
    fs.midi_to_audio(midi_file, wav_file)

    # Post-process WAV with pydub for genre/mood/duration
    audio = AudioSegment.from_wav(wav_file)

    # Duration trimming
    if duration > 0:
        audio = audio[: duration * 1000]

    # Mood filters
    if mood == "uplifting":
        audio = audio + 6
    elif mood == "melancholic":
        audio = audio.low_pass_filter(2000)
    elif mood == "energetic":
        audio = audio + 10
    elif mood == "calm":
        audio = audio - 4
    elif mood == "mysterious":
        audio = audio.high_pass_filter(5000)

    # Genre effects
    if genre == "cinematic":
        audio = audio.fade_in(2000).fade_out(2000)
    elif genre == "lofi":
        audio = audio.low_pass_filter(3000).apply_gain(-3)
    elif genre == "electronic":
        audio = audio + 4
    elif genre == "orchestral":
        audio = audio.fade_in(1000)
    elif genre == "ambient":
        audio = audio.fade_in(5000).fade_out(5000)

    # Save processed WAV
    audio.export(wav_file, format="wav")

    # Convert to MP3
    mp3_file = os.path.join(STATIC_DIR, f"{output_name}.mp3")
    audio.export(mp3_file, format="mp3")

    return wav_file, mp3_file

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/generate/text")
async def generate_text_music(
    text: str = Form(...),
    genre: str = Form(...),
    mood: str = Form(...),
    duration: int = Form(...),
    tempo: int = Form(...),
):
    try:
        abc_content, notes = generate_abc_from_text(text)
        wav_file, mp3_file = abc_to_wav_mp3(
            abc_content, "text_output", genre, mood, tempo, duration
        )
        return {
            "message": "Success",
            "wav_file": f"/static/{os.path.basename(wav_file)}",
            "mp3_file": f"/static/{os.path.basename(mp3_file)}",
            "notes": notes,
            "abc": abc_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating music: {str(e)}")


@app.post("/generate/image")
async def generate_image_music(
    file: UploadFile = File(...),
    genre: str = Form(...),
    mood: str = Form(...),
    duration: int = Form(...),
    tempo: int = Form(...),
):
    try:
        img = Image.open(file.file).convert("RGB")
        inputs = blip_processor(img, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        abc_content, notes = generate_abc_from_text(caption)
        wav_file, mp3_file = abc_to_wav_mp3(
            abc_content, "image_output", genre, mood, tempo, duration
        )
        return {
            "message": "Success",
            "caption": caption,
            "wav_file": f"/static/{os.path.basename(wav_file)}",
            "mp3_file": f"/static/{os.path.basename(mp3_file)}",
            "notes": notes,
            "abc": abc_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating music from image: {str(e)}")


@app.post("/generate/humming")
async def generate_humming_music(
    file: UploadFile = File(...),
    genre: str = Form(...),
    mood: str = Form(...),
    duration: int = Form(...),
    tempo: int = Form(...),
):
    try:
        # Save uploaded file
        humming_path = os.path.join(STATIC_DIR, "humming_input.wav")
        with open(humming_path, "wb") as f:
            f.write(await file.read())

        # Extract pitches
        y, sr = librosa.load(humming_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        melody_notes = []
        for i in range(pitches.shape[1]):
            idx = magnitudes[:, i].argmax()
            freq = pitches[idx, i]
            if freq > 0:
                n = note.Note()
                n.pitch.frequency = freq
                n.quarterLength = 0.25
                melody_notes.append(n)

        if not melody_notes:
            melody_notes = [note.Rest(quarterLength=1)]

        s = stream.Stream()
        for n in melody_notes:
            s.append(n)

        abc_content = "X:1\nL:1/8\nM:4/4\nK:C\n"
        abc_content += " ".join(
            n.nameWithOctave if isinstance(n, note.Note) else "z" for n in melody_notes
        )

        wav_file, mp3_file = abc_to_wav_mp3(
            abc_content, "humming_output", genre, mood, tempo, duration
        )

        notes_list = [n.nameWithOctave if isinstance(n, note.Note) else "Rest" for n in melody_notes]

        return {
            "message": "Success",
            "wav_file": f"/static/{os.path.basename(wav_file)}",
            "mp3_file": f"/static/{os.path.basename(mp3_file)}",
            "notes": notes_list,
            "abc": abc_content,
            "caption": "Humming converted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating music from humming: {str(e)}")

