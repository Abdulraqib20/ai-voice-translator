import os
import sys
import logging
import tempfile
import shutil
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/voice_translator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceTranslator:
    
    LANGUAGES = [
        'ar', 'zh', 'nl', 'en', 'fr', 'de', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'es', 'sv', 'tr', 'uk'
    ]
    
    def __init__(self, assemblyai_key: str, elevenlabs_key: str, voice_id: str):
        """Initialize the voice translator with API keys."""
        self.assemblyai_key = assemblyai_key
        self.elevenlabs_key = elevenlabs_key
        self.voice_id = voice_id
        self.temp_dir = Path(tempfile.mkdtemp())
        aai.settings.api_key = self.assemblyai_key
        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_key)

        # Validate API keys
        self.validate_api_keys()
        logger.info("VoiceTranslator initialized successfully")
    
    def __del__(self):
        """Cleanup temporary files on deletion."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary directory cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")
    
    def validate_api_keys(self):
        """Validate that all required API keys are present."""
        if not self.assemblyai_key:
            raise ValueError("Missing AssemblyAI API Key")
        if not self.elevenlabs_key:
            raise ValueError("Missing ElevenLabs API Key")
        if not self.voice_id:
            raise ValueError("Missing Voice ID")

    def voice_to_voice(self, audio_file):
        """Convert voice input to multiple translated voice outputs."""
        # Transcript speech
        transcript = self.transcribe_audio(audio_file)
        
        # Translate text
        list_translations = self.translate_text(transcript)
        generated_audio_paths = []

        # Generate speech from text
        for translation in list_translations:
            translated_audio_file_name = self.text_to_speech(translation)
            path = Path(translated_audio_file_name)
            generated_audio_paths.append(path)

        return (
            generated_audio_paths[0], generated_audio_paths[1], generated_audio_paths[2], 
            generated_audio_paths[3], generated_audio_paths[4], generated_audio_paths[5], 
            list_translations[0], list_translations[1], list_translations[2], 
            list_translations[3], list_translations[4], list_translations[5]
        )

    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text."""
        try:
            logger.info(f"Starting transcription for file: {audio_file}")
            transcriber = aai.Transcriber()

            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
            transcript = transcriber.transcribe(audio_file)
            if transcript.status == aai.TranscriptStatus.error:
                logger.error(f"Transcription error: {transcript.error}")
                raise gr.Error(f"Transcription failed: {transcript.error}")
            
            logger.info("Transcription completed successfully")
            return transcript.text

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise gr.Error(f"Transcription failed: {str(e)}")
    
    def translate_text(self, text: str) -> list:
        """Translate text to multiple languages."""
        list_translations = []
        target_languages = ['ar', 'zh', 'ru', 'fr', 'ja', 'ko']

        for lan in target_languages:
            translator = Translator(from_lang="en", to_lang=lan)
            translation = translator.translate(text)
            list_translations.append(translation)

        return list_translations

    def text_to_speech(self, text: str) -> str:
        """Convert text to speech using ElevenLabs API."""
        response = self.elevenlabs_client.text_to_speech.convert(
            voice_id=self.voice_id,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.8,
                style=0.5,
                use_speaker_boost=True,
            ),
        )

        save_file_path = str(self.temp_dir / f"{uuid.uuid4()}.mp3")

        # Writing the audio to a file
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        logger.info(f"Audio file saved successfully: {save_file_path}")
        return save_file_path

def create_demo():
    translator = VoiceTranslator(
        assemblyai_key=os.getenv("ASSEMBLYAI_API_KEY"),
        elevenlabs_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("VOICE_ID")
    )

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
        gr.Markdown(
            """
            # 🌍 AI-Powered Multilingual Voice Translator
            ### Transform your voice into multiple languages instantly

            Record your message in English, and receive translations in multiple languages.
            """
        )
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Record Audio (English)",
                    show_download_button=True,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        waveform_progress_color="#0066B4",
                        skip_length=2,
                        show_controls=False,
                    ),
                )
                with gr.Row():
                    submit = gr.Button("Translate", variant="primary")
                    status = gr.Textbox(label="Status", interactive=False)
                    clear_btn = gr.ClearButton([audio_input, status], value="Clear")

        with gr.Row():
            with gr.Group():
                ar_output = gr.Audio(label="Arabic", interactive=False)
                ar_text = gr.Markdown()

            with gr.Group():
                zh_output = gr.Audio(label="Chinese", interactive=False)
                zh_text = gr.Markdown()

            with gr.Group():
                ru_output = gr.Audio(label="Russian", interactive=False)
                ru_text = gr.Markdown()

        with gr.Row():
            with gr.Group():
                fr_output = gr.Audio(label="French", interactive=False)
                fr_text = gr.Markdown()

            with gr.Group():
                jp_output = gr.Audio(label="Japanese", interactive=False)
                jp_text = gr.Markdown()
            
            with gr.Group():
                ko_output = gr.Audio(label="Korean", interactive=False)
                ko_text = gr.Markdown()
                    
        output_components = [
            ar_output, zh_output, ru_output, fr_output, jp_output, ko_output,
            ar_text, zh_text, ru_text, fr_text, jp_text, ko_text
        ]
        
        submit.click(
            fn=translator.voice_to_voice, 
            inputs=audio_input, 
            outputs=output_components,
            show_progress=True
        )
        
        return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()