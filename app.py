import whisper
import gradio as gr

model = whisper.load_model("large")

def transcribe(audio):
    
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    ur_prob = probs["ur"]
    
    print(f"Probality of your speech being Urdu language: {ur_prob}")
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False,language="ur")
    result = whisper.decode(model, mel, options)
    return result.text
    
    
 
gr.Interface(
    title = '  اُردُو زبان میں خودکار تقریر کی شناخ', 
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()
