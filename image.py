import os
import re
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob
from music21 import converter, midi, stream, note, chord
from midi2audio import FluidSynth

# ----------------------------
# USER CONFIG: Instruments (multi-choice)
# ----------------------------
instrument_map = {
    "keyboard": 0,
    "guitar": 25,
    "drums": 118,
    "flute": 73
}

print("Available instruments:", ", ".join(instrument_map.keys()))
user_input = input("Choose instruments (comma-separated, e.g., keyboard,guitar,flute): ").lower()
user_choices = [i.strip() for i in user_input.split(",") if i.strip() in instrument_map]
if not user_choices:
    print("No valid instruments selected. Using default: keyboard")
    user_choices = ["keyboard"]
print("Selected instruments:", ", ".join(user_choices))

# ----------------------------
# IMAGE → CAPTION
# ----------------------------
image_file = "sunset.png"
if not os.path.exists(image_file):
    raise FileNotFoundError(f"The image file '{image_file}' was not found.")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
raw_image = Image.open(image_file).convert("RGB")

inputs = processor(raw_image, return_tensors="pt")
out = blip_model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Generated caption:", caption)

# ----------------------------
# CAPTION → EMOTION
# ----------------------------
def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        if "angry" in text.lower() or "storm" in text.lower():
            return "negative_angry"
        else:
            return "negative_sad"
    else:
        return "neutral"

emotion = detect_emotion(caption)
print("Detected emotion:", emotion)

# ----------------------------
# CAPTION + EMOTION → MUSIC (ABC)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained('sander-wood/text-to-music')
model = AutoModelForSeq2SeqLM.from_pretrained('sander-wood/text-to-music')

text_input = f"{caption}. Emotion: {emotion}. Instruments: {', '.join(user_choices)}."
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

# Clean ABC
tune_body = re.sub(r'[^A-Ga-gz,\s|]', '', tune_body)
abc_content = f"X:1\nL:1/8\nM:4/4\nK:C\n{tune_body}"
abc_file = "generated_tune.abc"
with open(abc_file, "w") as f:
    f.write(abc_content)
print("ABC file saved as", abc_file)

# ----------------------------
# ABC → MIDI
# ----------------------------
score = converter.parse(abc_content, format='abc')

# ----------------------------
# ADD DYNAMIC HARMONIES (no map)
# ----------------------------
melody = score.flatten().notes
harmony_stream = stream.Part()
for m_note in melody:
    if isinstance(m_note, note.Note):
        # Add a simple triad (root + major third + perfect fifth)
        triad = chord.Chord([m_note.pitch, m_note.pitch.transpose(4), m_note.pitch.transpose(7)])
        triad.quarterLength = m_note.quarterLength
        harmony_stream.append(triad)

melody_part = stream.Part()
for n in melody:
    melody_part.append(n)

final_score = stream.Score()
final_score.insert(0, melody_part)
final_score.insert(0, harmony_stream)

# ----------------------------
# MIDI FILE
# ----------------------------
midi_file = "generated_tune_harmony.mid"
mf = midi.translate.streamToMidiFile(final_score)
mf.open(midi_file, 'wb')
mf.write()
mf.close()
print("Harmony MIDI saved as", midi_file)

# ----------------------------
# MIDI → WAV
# ----------------------------
soundfont_path = "/Users/shreya/Downloads/GeneralUser GS 1.472/GeneralUser GS v1.472.sf2"
if not os.path.exists(soundfont_path):
    raise FileNotFoundError(f"SoundFont '{soundfont_path}' not found.")

fs = FluidSynth(soundfont_path)
wav_file = "generated_tune_harmony.wav"
fs.midi_to_audio(midi_file, wav_file)
print("Audio saved as", wav_file)
