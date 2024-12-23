import os
import sys
import logging
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
import tempfile
import shutil
from pydantic import ConfigDict
from typing import List, Tuple, Optional, Union

gr.__dict__.update(ConfigDict(arbitrary_types_allowed=True))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration keys
from config.appconfig import ASSEMBLYAI_API_KEY, ELEVENLABS_API_KEY, VOICE_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'logs/voice_translator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceTranslator:
    LANGUAGES = [
        ("ar", "Arabic"),
        ("zh", "Chinese"),
        ("nl", "Dutch"),
        ("en", "English"),
        ("fr", "French"),
        ("de", "German"),
        ("hi", "Hindi"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("pt", "Portuguese"),
        ("ru", "Russian"),
        ("es", "Spanish"),
        ("sv", "Swedish"),
        ("tr", "Turkish"),
        ("uk", "Ukrainian")
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
        if not self.assemblyai_key:
            raise ValueError("Missing AssemblyAI API Key")
        if not self.elevenlabs_key:
            raise ValueError("Missing ElevenLabs API Key")
        if not self.voice_id:
            raise ValueError("Missing Voice ID")

    def handle_audio_input(self, audio_data: Union[str, tuple]) -> str:
        """Handle different types of audio input and return a valid file path."""
        try:
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                temp_path = self.temp_dir / f"input_{uuid.uuid4()}.wav"
                gr.Audio.write_audio(str(temp_path), audio_array, sample_rate)
                return str(temp_path)
            elif isinstance(audio_data, str):
                return audio_data
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_data)}")
        except Exception as e:
            logger.error(f"Error handling audio input: {e}")
            raise gr.Error(f"Failed to process audio input: {str(e)}")

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
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

    def translate_text(self, text: str, selected_languages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Translate text to selected languages."""
        try:
            logger.info("Starting text translation")
            translations = []
            for lang_code, lang_name in selected_languages:
                try:
                    translator = Translator(from_lang="en", to_lang=lang_code)
                    translation = translator.translate(text)
                    translations.append((translation, lang_name))
                    logger.info(f"Translation to {lang_name} completed")
                except Exception as e:
                    logger.error(f"Error translating to {lang_name}: {e}")
                    translations.append((text, lang_name))  # Fallback to original text

            return translations

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise gr.Error(f"Translation failed: {str(e)}")

    def text_to_speech(self, text: str) -> str:
        """Convert text to speech using ElevenLabs."""
        try:
            logger.info("Starting text-to-speech conversion")
            output_path = self.temp_dir / f"{uuid.uuid4()}.mp3"

            response = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text[:500],  # Rate limit workaround
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.8,
                    style=0.5,
                    use_speaker_boost=True,
                ),
            )

            # Write the audio file in chunks
            with open(output_path, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)

            logger.info(f"Audio file saved successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            raise gr.Error(f"Text-to-speech conversion failed: {str(e)}")

    def process_voice(self, audio_input: Union[str, tuple], selected_languages: List[str]) -> List[Union[str, None]]:
        """Process voice input and return translated audio files and texts."""
        try:
            # Handle audio input and get valid file path
            audio_file = self.handle_audio_input(audio_input)
            
            # Convert selected language names to (code, name) tuples
            selected_langs = [(code, name) for code, name in self.LANGUAGES if name in selected_languages]
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_file)
            if not transcript:
                raise gr.Error("Transcription failed")

            # Translate text
            translations = self.translate_text(transcript, selected_langs)
            
            # Generate audio for each translation
            outputs = []
            for translation, _ in translations:
                try:
                    audio_path = self.text_to_speech(translation)
                    outputs.extend([audio_path, translation])
                except Exception as e:
                    logger.error(f"Error processing translation: {e}")
                    outputs.extend([None, translation])

            return outputs

        except Exception as e:
            logger.error(f"Voice processing error: {str(e)}")
            raise gr.Error(f"Voice processing failed: {str(e)}")

def create_gradio_interface(translator: VoiceTranslator):
    """Create and configure the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
        gr.Markdown(
            """
            # 🌍 AI-Powered Multilingual Voice Translator
            ### Transform your voice into multiple languages instantly

            Record your message in English, select at least 4 target languages, and receive translations.
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Record or Upload Audio (English)"
                )
                lang_selection = gr.CheckboxGroup(
                    choices=[lang_name for _, lang_name in VoiceTranslator.LANGUAGES],
                    label="Select at least 4 Target Languages",
                    interactive=True
                )
                with gr.Row():
                    submit_btn = gr.Button("Translate", variant="primary")
                    status = gr.Textbox(label="Status", interactive=False)

        # Create fixed output components
        output_boxes = []
        with gr.Row() as output_row:
            for _, lang_name in VoiceTranslator.LANGUAGES:
                with gr.Column(visible=False) as lang_column:
                    audio_out = gr.Audio(
                        label=f"{lang_name} Audio", 
                        interactive=False,
                        type="filepath"  # Ensure audio files are handled as files
                    )
                    text_out = gr.Textbox(label=f"{lang_name} Text", interactive=False)
                    output_boxes.append({
                        "column": lang_column,
                        "audio": audio_out,
                        "text": text_out,
                        "lang": lang_name
                    })

        def update_visible_outputs(selected_langs):
            """Update visibility of output components based on selected languages."""
            if len(selected_langs) < 4:
                raise gr.Error("Please select at least 4 languages")

            visibility_updates = []
            output_components = []
            
            for box in output_boxes:
                is_visible = box["lang"] in selected_langs
                visibility_updates.append(is_visible)
                
                if is_visible:
                    output_components.extend([box["audio"], box["text"]])

            return visibility_updates + output_components

        def process_inputs(audio, selected_langs):
            """Process audio input and return translations."""
            try:
                if not audio:
                    raise gr.Error("Please provide an audio input")
                if len(selected_langs) < 4:
                    raise gr.Error("Please select at least 4 languages")
                
                # Get translations from translator
                outputs = translator.process_voice(audio, selected_langs)
                
                # Calculate the total number of outputs needed
                total_languages = len(VoiceTranslator.LANGUAGES)  # Total number of possible languages
                
                # Create visibility updates for all possible languages
                visibility_updates = [box["lang"] in selected_langs for box in output_boxes]
                
                # Fill remaining outputs with None for languages not selected
                needed_outputs = total_languages * 2  # Each language needs 2 outputs (audio and text)
                current_outputs = len(outputs)
                padding = [None] * (needed_outputs - current_outputs)
                
                return visibility_updates + ["Translation completed successfully!"] + outputs + padding

            except Exception as e:
                # Handle errors with correct number of outputs
                error_msg = f"Error during translation: {str(e)}"
                visibility_updates = [box["lang"] in selected_langs for box in output_boxes]
                padding = [None] * (total_languages * 2)  # Full padding for all possible outputs
                return visibility_updates + [error_msg] + padding

        # Update visibility when languages are selected
        lang_selection.change(
            fn=update_visible_outputs,
            inputs=[lang_selection],
            outputs=[box["column"] for box in output_boxes] + 
                    [box["audio"] for box in output_boxes] + 
                    [box["text"] for box in output_boxes]
        )

        # Handle translation submission
        submit_btn.click(
            fn=process_inputs,
            inputs=[audio_input, lang_selection],
            outputs=[box["column"] for box in output_boxes] + 
                    [status] +  # Add status output
                    [box["audio"] for box in output_boxes] + 
                    [box["text"] for box in output_boxes],
            show_progress=True
        )

    return demo


if __name__ == "__main__":
    translator = VoiceTranslator(
        assemblyai_key=ASSEMBLYAI_API_KEY,
        elevenlabs_key=ELEVENLABS_API_KEY,
        voice_id=VOICE_ID
    )

    interface = create_gradio_interface(translator)
    interface.launch(share=True)
    