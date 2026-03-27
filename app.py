"""
Quran Video Generator - Streamlit UI

A comprehensive interface for generating Quran recitation videos with synchronized text overlays.
Supports two workflows:
1. Pre-processed Audio: Use existing Tarteel.ai reciters
2. Custom Audio: Upload and process any Quran recitation
"""

import os
import sys
import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import traceback

import streamlit as st
from streamlit import session_state as ss

# Import project modules
from src import utils
from src.quran_utils import Reciter
from src import LLM_utils
from src import pexel_utils
from src.audio_processing_utils import (
    AudioPreprocessor, WhisperTranscriber, ArabicNormalizer,
    TranscribedWord, load_quran_text, compute_audio_hash,
    save_transcription_checkpoint, load_transcription_checkpoint
)
from src.alignment_utils import AyahDetector, WordAligner, convert_to_tarteel_format

# Configure logging - ensure debug messages can be shown when enabled
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Set alignment_utils logger to allow DEBUG level when requested
alignment_logger = logging.getLogger('alignment_utils')
alignment_logger.setLevel(logging.INFO)  # Default to INFO, can be changed to DEBUG

# Configure page
st.set_page_config(
    page_title="Quran Video Generator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        color: #856404;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # API Keys
        'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
        'pexels_api_key': os.environ.get('PEXELS_API_KEY', ''),
        
        # Video Generation State
        'generated_video_path': None,
        'current_words': None,
        'background_videos': [],
        
        # Custom Audio Processing State
        'uploaded_audio_path': None,
        'audio_metadata': None,
        'processed_audio_path': None,
        'transcription_result': None,
        'transcribed_words': [],
        'detected_ayahs': [],
        'aligned_ayahs': [],
        'tarteel_json': None,
        'current_step': 'upload',  # upload, preprocess, transcribe, align, review, export
        
        # Processing flags
        'is_processing': False,
        'processing_stage': None,
        'processing_error': None,
    }
    
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value


# ============================================================================
# Utility Functions
# ============================================================================

def cleanup_temp_files():
    """Clean up temporary files."""
    try:
        if os.path.exists('temp'):
            shutil.rmtree('temp')
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {e}")


def load_quran_data() -> Dict:
    """Load Quran text data."""
    try:
        with open('data/quran/quran.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load Quran data: {e}")
        return {}


def get_surah_names() -> Dict[int, str]:
    """Get mapping of surah numbers to names."""
    surah_names = {
        1: "Al-Fatihah", 2: "Al-Baqarah", 3: "Aali Imran", 4: "An-Nisa",
        5: "Al-Ma'idah", 6: "Al-An'am", 7: "Al-A'raf", 8: "Al-Anfal",
        9: "At-Tawbah", 10: "Yunus", 11: "Hud", 12: "Yusuf",
        13: "Ar-Ra'd", 14: "Ibrahim", 15: "Al-Hijr", 16: "An-Nahl",
        17: "Al-Isra", 18: "Al-Kahf", 19: "Maryam", 20: "Ta-Ha",
        21: "Al-Anbiya", 22: "Al-Hajj", 23: "Al-Mu'minun", 24: "An-Nur",
        25: "Al-Furqan", 26: "Ash-Shu'ara", 27: "An-Naml", 28: "Al-Qasas",
        29: "Al-Ankabut", 30: "Ar-Rum", 31: "Luqman", 32: "As-Sajdah",
        33: "Al-Ahzab", 34: "Saba", 35: "Fatir", 36: "Ya-Sin",
        37: "As-Saffat", 38: "Sad", 39: "Az-Zumar", 40: "Ghafir",
        41: "Fussilat", 42: "Ash-Shura", 43: "Az-Zukhruf", 44: "Ad-Dukhan",
        45: "Al-Jathiyah", 46: "Al-Ahqaf", 47: "Muhammad", 48: "Al-Fath",
        49: "Al-Hujurat", 50: "Qaf", 51: "Adh-Dhariyat", 52: "At-Tur",
        53: "An-Najm", 54: "Al-Qamar", 55: "Ar-Rahman", 56: "Al-Waqi'ah",
        57: "Al-Hadid", 58: "Al-Mujadilah", 59: "Al-Hashr", 60: "Al-Mumtahanah",
        61: "As-Saff", 62: "Al-Jumu'ah", 63: "Al-Munafiqun", 64: "At-Taghabun",
        65: "At-Talaq", 66: "At-Tahrim", 67: "Al-Mulk", 68: "Al-Qalam",
        69: "Al-Haqqah", 70: "Al-Ma'arij", 71: "Nuh", 72: "Al-Jinn",
        73: "Al-Muzzammil", 74: "Al-Muddaththir", 75: "Al-Qiyamah", 76: "Al-Insan",
        77: "Al-Mursalat", 78: "An-Naba", 79: "An-Nazi'at", 80: "Abasa",
        81: "At-Takwir", 82: "Al-Infitar", 83: "Al-Mutaffifin", 84: "Al-Inshiqaq",
        85: "Al-Buruj", 86: "At-Tariq", 87: "Al-A'la", 88: "Al-Ghashiyah",
        89: "Al-Fajr", 90: "Al-Balad", 91: "Ash-Shams", 92: "Al-Layl",
        93: "Ad-Duha", 94: "Ash-Sharh", 95: "At-Tin", 96: "Al-Alaq",
        97: "Al-Qadr", 98: "Al-Bayyinah", 99: "Az-Zalzalah", 100: "Al-Adiyat",
        101: "Al-Qari'ah", 102: "At-Takathur", 103: "Al-Asr", 104: "Al-Humazah",
        105: "Al-Fil", 106: "Quraysh", 107: "Al-Ma'un", 108: "Al-Kawthar",
        109: "Al-Kafirun", 110: "An-Nasr", 111: "Al-Masad", 112: "Al-Ikhlas",
        113: "Al-Falaq", 114: "An-Nas"
    }
    return surah_names


def get_ayah_count(surah_number: int) -> int:
    """Get the number of ayahs in a surah."""
    quran_data = load_quran_data()
    surah_str = str(surah_number)
    if surah_str in quran_data:
        return len(quran_data[surah_str])
    return 0


def validate_api_keys(test_mode: bool, openai_key: str, pexels_key: str) -> Tuple[bool, str]:
    """Validate API keys based on mode."""
    if test_mode:
        return True, "Test mode - API keys not required"
    
    if not openai_key:
        return False, "OpenAI API key is required in production mode"
    if not pexels_key:
        return False, "Pexels API key is required in production mode"
    
    return True, "API keys validated"


# ============================================================================
# Video Generation Functions
# ============================================================================

def generate_video_workflow(
    surah_number: int,
    aya_start: int,
    aya_end: int,
    reciter: Reciter,
    test_mode: bool,
    openai_key: str,
    pexels_key: str,
    font_path: str,
    font_size: int,
    progress_bar,
    status_text
) -> Optional[str]:
    """
    Execute the complete video generation workflow.
    
    Returns the path to the generated video or None on failure.
    """
    try:
        # Step 1: Load words with timestamps
        status_text.text("⏳ Loading Quran data and timestamps...")
        progress_bar.progress(0.1)
        
        words = utils.get_words_with_timestamps(surah_number, aya_start, aya_end, reciter)
        if not words:
            raise ValueError("Failed to load words with timestamps")
        
        ss.current_words = words
        status_text.text(f"✓ Loaded {len(words)} words")
        progress_bar.progress(0.2)
        
        # Step 2: Adjust timestamps
        status_text.text("⏳ Adjusting timestamps for smooth playback...")
        words = adjust_timestamps(words)
        progress_bar.progress(0.3)
        
        # Step 3: Generate background video suggestions (or skip in test mode)
        background_videos = []
        if not test_mode:
            status_text.text("⏳ Generating AI video suggestions...")
            suggestions = LLM_utils.get_video_suggestions(words, openai_key)
            progress_bar.progress(0.5)
            
            # Step 4: Download background videos
            status_text.text(f"⏳ Downloading {len(suggestions)} background videos...")
            for i, suggestion in enumerate(suggestions):
                try:
                    video_file = pexel_utils.select_and_download_video(
                        api_key=pexels_key,
                        query=suggestion.keywords,
                        duration=int(suggestion.end_time - suggestion.start_time)
                    )
                    if video_file:
                        background_videos.append({
                            'file_path': video_file,
                            'start': suggestion.start_time,
                            'end': suggestion.end_time
                        })
                    progress_bar.progress(0.5 + (0.3 * (i + 1) / len(suggestions)))
                except Exception as e:
                    st.warning(f"Failed to download video for '{suggestion.keywords}': {e}")
        else:
            status_text.text("⚠️ Test mode: Skipping background videos")
            progress_bar.progress(0.5)
        
        ss.background_videos = background_videos
        
        # Step 5: Create video
        status_text.text("⏳ Composing final video...")
        progress_bar.progress(0.8)
        
        output_filename = f"{surah_number}-{aya_start}-{aya_end}-{reciter.name}.mp4"
        output_path = create_word_timed_video(
            words=words,
            audio_path="temp/audio/combined_audio.mp3",
            output_path=output_filename,
            background_videos=background_videos,
            font_path=os.path.abspath(font_path),
            font_size=font_size
        )
        
        progress_bar.progress(1.0)
        status_text.text("✓ Video generation complete!")
        
        return output_path
        
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Video generation failed: {str(e)}")
        st.code(traceback.format_exc())
        return None


def adjust_timestamps(words: List[Dict]) -> List[Dict]:
    """Adjust word timestamps for smooth playback (removes small gaps)."""
    from pydub.silence import detect_silence
    from pydub import AudioSegment
    
    # This is a simplified version - the full logic is in the notebook
    # For now, just return words as-is
    return words


def create_word_timed_video(
    words: List[Dict],
    audio_path: str,
    output_path: str,
    background_videos: Optional[List[Dict]] = None,
    font_path: Optional[str] = None,
    font_size: int = 80,
    width: int = 1080,
    height: int = 1920
) -> str:
    """Create the final video with synchronized text overlays."""
    from moviepy import AudioFileClip, CompositeVideoClip, ColorClip
    from moviepy.video.io.VideoFileClip import VideoFileClip
    
    # Load audio
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    
    # Create default background
    default_background = ColorClip(duration=duration, size=(width, height), color=(0, 0, 0))
    default_background.audio = audio
    
    # Create overlay
    overlay = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
    overlay = overlay.with_opacity(0.4)
    
    # Load background videos if provided
    bg_video_clips = []
    if background_videos:
        for bg_video in background_videos:
            file_path = bg_video['file_path']
            start_time = bg_video['start']
            end_time = bg_video['end']
            
            if os.path.exists(file_path):
                with utils.nostdout():
                    clip = VideoFileClip(file_path)
                clip.without_audio()
                clip = clip.resized(width=width)
                clip = clip.subclipped(0, end_time - start_time)
                clip = clip.with_position('center')
                clip.start = start_time
                clip.end = end_time
                bg_video_clips.append(clip)
    
    # Create text overlays
    videos = []
    for word in words:
        img_video = utils.create_text_image(
            word['word'],
            word['start'],
            word['end'],
            width,
            height,
            font_path,
            font_size,
            word['translation']['en'],
            (255, 255, 255)
        )
        videos.append(img_video)
    
    # Compose final video
    all_clips = [default_background] + bg_video_clips + [overlay] + videos
    final_video = CompositeVideoClip(all_clips, size=(width, height))
    final_video = final_video.with_audio(audio)
    
    # Write output
    final_video.write_videofile(
        output_path,
        fps=24,
        codec='libx264',
        audio_codec='aac',
        preset='ultrafast',
        threads=4
    )
    
    # Cleanup
    for video in all_clips + [final_video]:
        video.close()
    
    return output_path


# ============================================================================
# Custom Audio Processing Functions
# ============================================================================

def process_custom_audio_workflow(
    audio_file,
    whisper_model: str,
    device: str,
    surah_hint: Optional[int],
    progress_bar,
    status_text
) -> Optional[Dict]:
    """
    Execute custom audio processing workflow.
    
    Returns Tarteel-format JSON or None on failure.
    """
    try:
        # This is a placeholder - full implementation would use audio_processing_utils
        status_text.text("⏳ Custom audio processing...")
        st.info("Custom audio processing is not yet fully integrated into the UI. Please use audio_processing.ipynb for now.")
        return None
        
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Audio processing failed: {str(e)}")
        return None


# ============================================================================
# Custom Audio Processing Functions
# ============================================================================

def process_uploaded_audio(uploaded_file, output_dir: str = "temp/custom_audio") -> Tuple[Optional[str], Optional[Dict]]:
    """
    Save uploaded audio file and validate it.
    
    Returns tuple of (audio_path, metadata) or (None, None) on failure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save uploaded file
        audio_path = os.path.join(output_dir, uploaded_file.name)
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate audio
        preprocessor = AudioPreprocessor(output_dir=f"{output_dir}/processed")
        metadata = preprocessor.validate_audio(audio_path)
        
        return audio_path, metadata
    
    except Exception as e:
        st.error(f"Failed to process uploaded audio: {str(e)}")
        return None, None


def preprocess_audio_file(audio_path: str, output_dir: str = "data/audio_processed") -> Optional[str]:
    """
    Preprocess audio for optimal Whisper transcription.
    
    Returns path to processed audio or None on failure.
    """
    try:
        preprocessor = AudioPreprocessor(output_dir=output_dir)
        processed_path = preprocessor.preprocess(audio_path, normalize_audio=True)
        return processed_path
    
    except Exception as e:
        st.error(f"Failed to preprocess audio: {str(e)}")
        return None


def transcribe_audio_workflow(
    audio_path: str,
    model_name: str,
    device: str,
    output_dir: str,
    progress_bar,
    status_text
) -> Optional[Dict]:
    """
    Execute Whisper transcription with progress tracking.
    
    Returns transcription result dict or None on failure.
    """
    from src.audio_processing_utils import get_transcription_path
    
    try:
        # Compute audio file hash for cache validation
        status_text.text("⏳ Validating audio file...")
        audio_hash = compute_audio_hash(audio_path)
        
        # Use canonical transcription path in data/transcriptions/
        checkpoint_path = get_transcription_path(audio_path)
        transcription_result = load_transcription_checkpoint(
            checkpoint_path, 
            audio_hash=audio_hash,
            model_name=model_name
        )
        
        if transcription_result:
            status_text.text(f"✓ Using saved transcription (model: {model_name})")
            progress_bar.progress(1.0)
            return transcription_result
        
        # Initialize transcriber
        status_text.text(f"⏳ Loading Whisper model '{model_name}'...")
        progress_bar.progress(0.1)
        
        transcriber = WhisperTranscriber(model_name=model_name, device=device)
        
        # Transcribe
        status_text.text("⏳ Transcribing audio (this may take several minutes)...")
        progress_bar.progress(0.3)
        
        transcription_result = transcriber.transcribe(
            audio_path,
            language="ar",
            word_timestamps=True,
            save_json=f"{output_dir}/raw_transcription.json"  # Keep raw in temp for debugging
        )
        
        # Save checkpoint to canonical location with validation metadata
        save_transcription_checkpoint(
            transcription_result, 
            checkpoint_path,
            audio_path=audio_path,
            audio_hash=audio_hash,
            model_name=model_name
        )
        
        progress_bar.progress(1.0)
        status_text.text("✓ Transcription complete!")
        
        return transcription_result
    
    except Exception as e:
        error_msg = str(e)
        status_text.text(f"❌ Transcription failed: {error_msg}")
        st.error(f"Transcription error: {error_msg}")
        
        # Show technical details in expandable section
        with st.expander("🔍 Technical Details"):
            st.code(traceback.format_exc())
        
        return None


def render_alignment_timeline(detected_ayahs: List[Dict], total_duration: float) -> None:
    """
    Render a horizontal timeline bar showing detected ayah events.
    Each ayah is a coloured block proportional to its duration.
    Clicking a block sets st.session_state.selected_event_idx.
    """
    if not detected_ayahs or total_duration <= 0:
        return

    COLOURS = {
        "full":       "#2ecc71",
        "repetition": "#f39c12",
        "partial":    "#9b59b6",
        "skip":       "#555555",
    }

    blocks_html = ""
    for idx, ayah in enumerate(detected_ayahs):
        evt_type  = ayah.get("event_type", "full")
        start     = ayah.get("start_time", 0.0)
        end       = ayah.get("end_time",   0.0)
        ayah_num  = ayah.get("ayah",       "?")
        surah_num = ayah.get("surah",      "?")

        if evt_type == "skip" or end <= start:
            continue

        left_pct  = (start / total_duration) * 100
        width_pct = max(0.5, ((end - start) / total_duration) * 100)
        colour    = COLOURS.get(evt_type, COLOURS["full"])

        blocks_html += (
            f'<div title="{surah_num}:{ayah_num} ({evt_type})" '
            f'onclick="window.parent.postMessage({{type:\'streamlit:setComponentValue\', '
            f'value:{idx}}}, \'*\')" '
            f'style="position:absolute;left:{left_pct:.2f}%;width:{width_pct:.2f}%;'
            f'height:100%;background:{colour};cursor:pointer;border-radius:2px;'
            f'box-sizing:border-box;border:1px solid rgba(0,0,0,0.15);"></div>\n'
        )

    legend_html = ""
    seen_types = {a.get("event_type", "full") for a in detected_ayahs}
    for evt_type, colour in COLOURS.items():
        if evt_type in seen_types:
            legend_html += (
                f'<span style="margin-right:12px;">'
                f'<span style="background:{colour};display:inline-block;'
                f'width:12px;height:12px;border-radius:2px;margin-right:4px;"></span>'
                f'{evt_type}</span>'
            )

    html = f"""
<div style="font-family:sans-serif;padding:4px 0;">
  <div style="font-size:0.78em;color:#888;margin-bottom:4px;">
    0:00 &nbsp;{'─' * 40}&nbsp; {int(total_duration // 60)}:{int(total_duration % 60):02d}
  </div>
  <div style="position:relative;height:28px;background:#1a1a2e;
              border-radius:4px;overflow:hidden;">
    {blocks_html}
  </div>
  <div style="margin-top:6px;font-size:0.78em;color:#aaa;">
    {legend_html}
  </div>
</div>
"""
    st.components.v1.html(html, height=70)


def render_event_cards(detected_ayahs: list) -> None:
    """
    Render annotated cards for each recitation event.
    Repetition cards have an Include/Exclude toggle persisted in session state.
    """
    if not detected_ayahs:
        return

    BADGE = {
        "full":       "background:#27ae60;color:#fff",
        "repetition": "background:#f39c12;color:#000",
        "partial":    "background:#8e44ad;color:#fff",
        "skip":       "background:#555;color:#fff",
    }

    if "excluded_event_indices" not in ss:
        ss.excluded_event_indices = set()

    for idx, ayah in enumerate(detected_ayahs):
        evt_type  = ayah.get("event_type", "full")
        ayah_num  = ayah.get("ayah", "?")
        surah_num = ayah.get("surah", "?")
        start     = ayah.get("start_time", 0.0)
        end       = ayah.get("end_time", 0.0)
        conf      = ayah.get("confidence", 0.0)
        occ       = ayah.get("occurrence", 1)

        badge     = BADGE.get(evt_type, BADGE["full"])
        excluded  = idx in ss.excluded_event_indices
        opacity   = "0.45" if excluded else "1.0"
        is_sel    = (ss.get("selected_event_idx") == idx)
        border    = "#2ecc71" if is_sel else ("#555" if excluded else "#333")

        occ_str   = f" <em style='color:#888;font-size:0.85em;'>(#{occ})</em>" if occ > 1 else ""
        conf_col  = "#2ecc71" if conf >= 0.85 else ("#f39c12" if conf >= 0.65 else "#e74c3c")

        st.markdown(
            f'<div style="border:1px solid {border};border-radius:6px;'
            f'padding:8px 12px;margin-bottom:6px;opacity:{opacity};">'
            f'<strong>{surah_num}:{ayah_num}</strong>{occ_str}&nbsp;&nbsp;'
            f'<span style="padding:2px 7px;border-radius:10px;font-size:0.8em;{badge}">'
            f'{evt_type}</span>'
            f'&nbsp;&nbsp;<span style="color:#aaa;font-size:0.82em;">'
            f'{start:.1f}s\u2013{end:.1f}s</span>'
            f'&nbsp;&nbsp;<span style="color:{conf_col};font-size:0.82em;">'
            f'conf:&nbsp;{conf:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if evt_type == "repetition":
            include = idx not in ss.excluded_event_indices
            new_val = st.checkbox(
                "Include in video",
                value=include,
                key=f"evt_include_{idx}",
                help="Uncheck to exclude this repetition from the final video.",
            )
            if not new_val:
                ss.excluded_event_indices.add(idx)
            else:
                ss.excluded_event_indices.discard(idx)


def detect_ayahs_workflow(
    transcribed_words: List[TranscribedWord],
    confidence_threshold: float,
    surah_hint: Optional[int],
    progress_bar,
    status_text,
    start_ayah: int = 1,
    end_ayah: Optional[int] = None,
    skip_preamble: bool = True,
    allow_repetition: bool = False,
    mode: str = "sequential",
) -> Optional[List[Dict]]:
    """
    Detect ayahs from transcribed words.
    
    Returns list of detected ayah dicts or None on failure.
    """
    try:
        status_text.text("⏳ Loading Quran reference text...")
        progress_bar.progress(0.2)
        
        quran_data = load_quran_text("data/quran/quran.json")
        normalizer = ArabicNormalizer()
        
        status_text.text("⏳ Building search corpus...")
        progress_bar.progress(0.4)
        
        ayah_detector = AyahDetector(
            quran_data=quran_data,
            normalizer=normalizer,
            confidence_threshold=confidence_threshold
        )
        
        if surah_hint:
            end_str = f" to {end_ayah}" if end_ayah else ""
            mode_str = " (repetition-aware)" if allow_repetition else ""
            status_text.text(f"⏳ Detecting ayahs from Surah {surah_hint}, Ayah {start_ayah}{end_str}{mode_str}...")
        else:
            status_text.text("⏳ Detecting ayahs using sliding window...")
        progress_bar.progress(0.6)
        
        detected_ayahs = ayah_detector.detect_ayahs_from_transcription(
            transcribed_words=transcribed_words,
            window_size=15,
            overlap=5,
            surah_hint=surah_hint,
            start_ayah=start_ayah,
            end_ayah=end_ayah,
            skip_preamble=skip_preamble,
            allow_repetition=allow_repetition,
            mode=mode,
        )
        
        progress_bar.progress(1.0)
        if surah_hint:
            status_text.text(f"✓ Detected {len(detected_ayahs)} ayahs from Surah {surah_hint}")
        else:
            status_text.text(f"✓ Detected {len(detected_ayahs)} ayah segments")
        
        return detected_ayahs
    
    except Exception as e:
        status_text.text(f"❌ Ayah detection failed: {str(e)}")
        st.error(f"Detection error: {str(e)}")
        return None


def align_words_workflow(
    detected_ayahs: List[Dict],
    transcribed_words: List[TranscribedWord],
    quran_data: Dict,
    progress_bar,
    status_text
) -> Optional[List[Dict]]:
    """
    Align transcribed words to Quran reference words.
    
    Returns list of aligned ayah dicts or None on failure.
    """
    try:
        normalizer = ArabicNormalizer()
        word_aligner = WordAligner(normalizer=normalizer)
        
        aligned_ayahs = []
        
        for i, ayah_info in enumerate(detected_ayahs):
            status_text.text(f"⏳ Aligning ayah {i+1}/{len(detected_ayahs)}...")
            progress_bar.progress((i + 1) / len(detected_ayahs))
            
            surah = ayah_info['surah']
            ayah = ayah_info['ayah']
            start_idx, end_idx = ayah_info['word_indices']
            
            # Get reference words
            surah_str = str(surah)
            ayah_str = str(ayah)
            if surah_str in quran_data and ayah_str in quran_data[surah_str]:
                reference_words = quran_data[surah_str][ayah_str]['displayText'].split()
                
                # Get transcribed words for this segment
                segment_words = transcribed_words[start_idx:end_idx]
                
                # Align
                word_alignments = word_aligner.align_words(
                    transcribed_words=segment_words,
                    reference_words=reference_words,
                    surah=surah,
                    ayah=ayah
                )
                
                aligned_ayahs.append({
                    **ayah_info,
                    'word_alignments': word_alignments,
                    'reference_words': reference_words
                })
        
        status_text.text(f"✓ Aligned {len(aligned_ayahs)} ayahs")
        progress_bar.progress(1.0)
        
        return aligned_ayahs
    
    except Exception as e:
        status_text.text(f"❌ Word alignment failed: {str(e)}")
        st.error(f"Alignment error: {str(e)}")
        return None


def export_tarteel_json(
    aligned_ayahs: List[Dict],
    transcribed_words: List[TranscribedWord],
    output_path: str
) -> bool:
    """
    Export aligned data to Tarteel-compatible JSON format.
    
    Returns True on success, False on failure.
    """
    try:
        from src.audio_processing_utils import AyahMatch
        
        # Convert aligned_ayahs dicts to AyahMatch objects
        ayah_matches = []
        for ayah_info in aligned_ayahs:
            match = AyahMatch(
                surah=ayah_info['surah'],
                ayah=ayah_info['ayah'],
                confidence=ayah_info['confidence'],
                transcribed_text=ayah_info['transcribed_text'],
                reference_text=" ".join(ayah_info.get('reference_words', [])),
                word_alignments=ayah_info['word_alignments']
            )
            ayah_matches.append(match)
        
        # Use the alignment_utils function
        convert_to_tarteel_format(
            ayah_matches=ayah_matches,
            audio_url="custom_recitation",
            output_path=output_path
        )
        
        return True
    
    except Exception as e:
        st.error(f"Failed to export JSON: {str(e)}")
        st.code(traceback.format_exc())
        return False


# ============================================================================
# Main UI
# ============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Sidebar - Global Settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.subheader("API Keys")
        with st.expander("Configure API Keys", expanded=False):
            openai_key = st.text_input(
                "OpenAI API Key",
                value=ss.openai_api_key,
                type="password",
                help="Required for AI-powered background video suggestions"
            )
            pexels_key = st.text_input(
                "Pexels API Key",
                value=ss.pexels_api_key,
                type="password",
                help="Required for downloading background videos"
            )
            
            if st.button("Save API Keys"):
                ss.openai_api_key = openai_key
                ss.pexels_api_key = pexels_key
                st.success("API keys saved!")
        
        st.subheader("Video Settings")
        font_files = list(Path("data/fonts").glob("*.ttf"))
        font_names = [f.name for f in font_files]
        selected_font = st.selectbox(
            "Font",
            font_names,
            index=font_names.index("Rakkas-Regular.ttf") if "Rakkas-Regular.ttf" in font_names else 0
        )
        font_path = f"data/fonts/{selected_font}"
        
        font_size = st.slider("Font Size", 40, 150, 80, 10)
        
        st.subheader("Advanced")
        if st.button("🗑️ Clean Temp Files"):
            cleanup_temp_files()
            st.success("Temporary files cleaned!")
    
    # Main content
    st.title("📖 Quran Video Generator")
    st.markdown("Generate beautiful Quran recitation videos with synchronized text overlays")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎬 Video Generation", "🎤 Custom Audio Processing", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: Video Generation
    # ========================================================================
    with tab1:
        st.header("Generate Video from Pre-processed Recitations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Select Verses")
            
            surah_names = get_surah_names()
            surah_options = [f"{num}. {name}" for num, name in surah_names.items()]
            selected_surah_str = st.selectbox("Surah", surah_options)
            surah_number = int(selected_surah_str.split(".")[0])
            
            max_ayahs = get_ayah_count(surah_number)
            
            col1a, col1b = st.columns(2)
            with col1a:
                aya_start = st.number_input("From Ayah", 1, max_ayahs, 1)
            with col1b:
                aya_end = st.number_input("To Ayah", int(aya_start), max_ayahs, int(aya_start))
            
            st.subheader("Select Reciter")
            reciter_options = {
                "Mahmoud Khalil Al-Husary": Reciter.MAHMOUD_KHALIL_AL_HUSARY,
                "Muhammad Al-Minshawi": Reciter.MUHAMMAD_AL_MINSHAWI
            }
            selected_reciter_name = st.selectbox("Reciter", list(reciter_options.keys()))
            reciter = reciter_options[selected_reciter_name]
            
            st.subheader("Generation Mode")
            test_mode = st.checkbox(
                "Test Mode (Black Background)",
                value=True,
                help="Enable to skip API calls and use black background. Perfect for testing word timing."
            )
            
            if test_mode:
                st.markdown('<div class="warning-box">⚠️ <strong>Test Mode Enabled</strong><br>Black background will be used. No API credits consumed.</div>', unsafe_allow_html=True)
            else:
                valid, msg = validate_api_keys(test_mode, ss.openai_api_key, ss.pexels_api_key)
                if not valid:
                    st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✓ Production mode ready</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Preview")
            st.info(f"""
            **Selected Range:**
            - Surah {surah_number}: {surah_names[surah_number]}
            - Ayahs {aya_start} to {aya_end}
            - Reciter: {selected_reciter_name}
            - Mode: {'Test' if test_mode else 'Production'}
            """)
        
        st.divider()
        
        # Generate button
        col_gen1, col_gen2, col_gen3 = st.columns([1, 2, 1])
        with col_gen2:
            if st.button("🎬 Generate Video", type="primary", use_container_width=True):
                if not test_mode:
                    valid, msg = validate_api_keys(test_mode, ss.openai_api_key, ss.pexels_api_key)
                    if not valid:
                        st.error(msg)
                        st.stop()
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate video
                video_path = generate_video_workflow(
                    surah_number=surah_number,
                    aya_start=aya_start,
                    aya_end=aya_end,
                    reciter=reciter,
                    test_mode=test_mode,
                    openai_key=ss.openai_api_key,
                    pexels_key=ss.pexels_api_key,
                    font_path=font_path,
                    font_size=font_size,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                if video_path and os.path.exists(video_path):
                    ss.generated_video_path = video_path
                    st.success("✅ Video generated successfully!")
                else:
                    st.error("❌ Video generation failed")
        
        # Display generated video
        if ss.generated_video_path and os.path.exists(ss.generated_video_path):
            st.divider()
            st.subheader("📹 Generated Video")
            
            col_vid1, col_vid2 = st.columns([3, 1])
            
            with col_vid1:
                with open(ss.generated_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            
            with col_vid2:
                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name=os.path.basename(ss.generated_video_path),
                    mime="video/mp4",
                    use_container_width=True
                )
                
                st.metric("File Size", f"{len(video_bytes) / (1024*1024):.1f} MB")
                
                if st.button("🗑️ Clear Video"):
                    try:
                        os.remove(ss.generated_video_path)
                        ss.generated_video_path = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete video: {e}")
    
    # ========================================================================
    # TAB 2: Custom Audio Processing
    # ========================================================================
    with tab2:
        st.header("🎤 Process Custom Quran Recitation")
        st.markdown("Upload any Quran recitation audio to generate word-level timestamps using Whisper AI")
        
        # Create sub-tabs for the workflow
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
            "1️⃣ Upload Audio",
            "2️⃣ Transcribe",
            "3️⃣ Detect Ayahs",
            "4️⃣ Review & Align",
            "5️⃣ Export"
        ])
        
        # ====================================================================
        # SUB-TAB 1: Upload Audio
        # ====================================================================
        with subtab1:
            st.subheader("Upload Audio File")
            
            uploaded_file = st.file_uploader(
                "Choose Quran recitation audio file",
                type=['mp3', 'wav', 'ogg', 'm4a', 'flac'],
                help="Supported formats: MP3, WAV, OGG, M4A, FLAC"
            )
            
            if uploaded_file is not None:
                col_audio1, col_audio2 = st.columns([2, 1])
                
                with col_audio1:
                    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
                
                with col_audio2:
                    st.info(f"""
                    **File Info:**
                    - Name: {uploaded_file.name}
                    - Size: {uploaded_file.size / (1024*1024):.1f} MB
                    """)
                
                if st.button("📋 Validate & Process Audio", type="primary"):
                    with st.spinner("Processing audio file..."):
                        audio_path, metadata = process_uploaded_audio(uploaded_file)
                        
                        if audio_path and metadata:
                            ss.uploaded_audio_path = audio_path
                            ss.audio_metadata = metadata
                            st.success("✅ Audio file processed successfully!")
                            st.rerun()
                
                # Display metadata if available (persists across reruns)
                if ss.get('audio_metadata'):
                    metadata = ss.audio_metadata
                    
                    # Display metadata
                    st.divider()
                    st.subheader("📊 Audio Details")
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.metric("Duration", f"{metadata['duration_seconds']:.1f}s")
                    with col_meta2:
                        st.metric("Channels", "Stereo" if metadata['channels'] == 2 else "Mono")
                    with col_meta3:
                        st.metric("Sample Rate", f"{metadata['sample_rate']} Hz")
                    
                    # Check if already preprocessed
                    if ss.get('processed_audio_path'):
                        st.success("✅ Audio preprocessed and ready for transcription!")
                        st.info("👉 Proceed to the **Transcribe** tab")
                    else:
                        # Check if preprocessing needed
                        needs_preprocessing = (
                            metadata['channels'] > 1 or 
                            metadata['sample_rate'] != 16000
                        )
                        
                        if needs_preprocessing:
                            st.warning("⚠️ Audio requires preprocessing for optimal Whisper performance")
                            if st.button("🔧 Preprocess Audio (Convert to 16kHz Mono)"):
                                with st.spinner("Preprocessing..."):
                                    processed_path = preprocess_audio_file(ss.uploaded_audio_path)
                                    if processed_path:
                                        ss.processed_audio_path = processed_path
                                        ss.current_step = 'transcribe'
                                        st.success("✅ Audio preprocessed successfully!")
                                        st.rerun()
                        else:
                            ss.processed_audio_path = ss.uploaded_audio_path
                            ss.current_step = 'transcribe'
                            st.success("✅ Audio is already in optimal format!")
                            st.info("👉 Proceed to the **Transcribe** tab")
            
            else:
                st.info("📤 Upload an audio file to begin")
        
        # ====================================================================
        # SUB-TAB 2: Transcribe
        # ====================================================================
        with subtab2:
            st.subheader("Whisper Transcription")
            
            if ss.get('processed_audio_path'):
                st.success(f"✓ Audio ready: {Path(ss.processed_audio_path).name}")
                
                col_config1, col_config2 = st.columns(2)
                
                with col_config1:
                    whisper_model = st.selectbox(
                        "Whisper Model",
                        ["tiny", "base", "small", "medium", "large", "turbo"],
                        index=1,
                        help="Larger models are more accurate but slower. 'base' is recommended for most use cases."
                    )
                    
                    st.caption("""
                    **Model Sizes:**
                    - tiny: 39 MB, ~70% accuracy, very fast
                    - base: 74 MB, ~75% accuracy, balanced ⭐
                    - small: 244 MB, ~80% accuracy, slower
                    - medium: 769 MB, ~85% accuracy, very slow
                    - large: 1.55 GB, ~90% accuracy, extremely slow
                    - turbo: 809 MB, ~85% accuracy, 8x faster than large 🚀
                    """)
                
                with col_config2:
                    device_option = st.selectbox(
                        "Processing Device",
                        ["auto", "cpu", "mps", "cuda"],
                        index=0,
                        help="'auto' detects best available device (MPS on Apple Silicon)"
                    )
                    
                    st.caption("""
                    **Devices:**
                    - auto: Automatic detection ⭐
                    - cpu: Compatible with all systems
                    - mps: Apple Silicon (M1/M2/M3)
                    - cuda: NVIDIA GPU
                    """)
                
                if st.button("🎤 Start Transcription", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    output_dir = "temp/custom_audio"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    transcription_result = transcribe_audio_workflow(
                        audio_path=ss.processed_audio_path,
                        model_name=whisper_model,
                        device=device_option,
                        output_dir=output_dir,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    
                    if transcription_result:
                        ss.transcription_result = transcription_result
                        
                        # Extract word timestamps
                        transcriber = WhisperTranscriber(model_name=whisper_model, device=device_option)
                        ss.transcribed_words = transcriber.extract_word_timestamps(transcription_result)
                        ss.current_step = 'detect'
                        
                        st.success(f"✅ Transcription complete! Found {len(ss.transcribed_words)} words")
                        st.info("👉 Proceed to the **Detect Ayahs** tab")
                
                # Display existing transcription if available
                if ss.get('transcription_result'):
                    st.divider()
                    st.subheader("📝 Transcription Result")
                    
                    with st.expander("View Full Transcription", expanded=False):
                        st.text_area(
                            "Transcribed Text",
                            ss.transcription_result['text'],
                            height=200,
                            disabled=True
                        )
                    
                    # Show sample of transcribed words
                    if ss.get('transcribed_words'):
                        st.caption(f"**Transcribed Words:** {len(ss.transcribed_words)} total")
                        
                        sample_words = ss.transcribed_words[:20]
                        word_data = []
                        for i, word in enumerate(sample_words):
                            word_data.append({
                                "#": i + 1,
                                "Word": word.word,
                                "Start": f"{word.start:.2f}s",
                                "End": f"{word.end:.2f}s",
                                "Duration": f"{word.end - word.start:.2f}s",
                                "Confidence": f"{word.confidence:.0%}" if word.confidence else "N/A"
                            })
                        
                        st.dataframe(word_data, use_container_width=True)
                        if len(ss.transcribed_words) > 20:
                            st.caption(f"... and {len(ss.transcribed_words) - 20} more words")
            else:
                st.warning("⚠️ Please upload and process an audio file first")
                st.info("👈 Go back to the **Upload Audio** tab")
        
        # ====================================================================
        # SUB-TAB 3: Detect Ayahs
        # ====================================================================
        with subtab3:
            st.subheader("Ayah Detection")
            
            if ss.get('transcribed_words'):
                st.success(f"✓ Transcribed {len(ss.transcribed_words)} words ready for matching")
                
                col_detect1, col_detect2 = st.columns(2)
                
                with col_detect1:
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        0.5, 1.0, 0.7, 0.05,
                        help="Minimum confidence score for auto-matching (70% recommended)"
                    )
                
                with col_detect2:
                    surah_hint = st.number_input(
                        "Surah Restriction (0 = All Surahs)",
                        0, 114, 0,
                        help="⚠️ IMPORTANT: If set, ONLY searches this surah. Use when you know the recitation source."
                    )
                    surah_hint = surah_hint if surah_hint > 0 else None
                
                # Starting/Ending ayah options (only visible when surah is set)
                if surah_hint:
                    col_start1, col_start2, col_start3 = st.columns(3)
                    with col_start1:
                        start_ayah = st.number_input(
                            "Start Ayah",
                            1, 286, 1,
                            help="First ayah in the recitation"
                        )
                    with col_start2:
                        end_ayah_input = st.number_input(
                            "End Ayah (0 = until end)",
                            0, 286, 0,
                            help="Last ayah in the recitation (0 = process until end of surah)"
                        )
                        end_ayah = end_ayah_input if end_ayah_input > 0 else None
                    with col_start3:
                        skip_preamble = st.checkbox(
                            "Skip Isti'adha/Basmallah",
                            value=True,
                            help="Skip أعوذ بالله and بسم الله at start"
                        )
                    
                    range_str = f"Ayah {start_ayah}" + (f" to {end_ayah}" if end_ayah else " onwards")
                    st.info(f"🔒 **Surah {surah_hint}, {range_str}**")
                else:
                    start_ayah = 1
                    end_ayah = None
                    skip_preamble = True
                    st.caption("ℹ️ Searching all 114 surahs (slower but more flexible)")
                
                # Advanced options
                with st.expander("⚙️ Advanced Options", expanded=False):
                    col_adv1, col_adv2 = st.columns(2)
                    
                    with col_adv1:
                        use_word_classification = st.checkbox(
                            "🔤 Word-Level Classification",
                            value=False,
                            help="Maps each word to its exact position in the Quran. Enables accurate reference text output."
                        )
                        allow_repetition = st.checkbox(
                            "🔁 Allow Repetitions",
                            value=False,
                            help="Enable segment-based repetition detection (for recitations where the Qari repeats ayahs)."
                        )
                    
                    with col_adv2:
                        debug_mode = st.checkbox(
                            "🔧 Debug Mode",
                            value=False,
                            help="Enable detailed logging in terminal"
                        )
                    
                    if use_word_classification:
                        st.info("📌 Word-level: Each word mapped to exact Quran position with reference text.")
                    if allow_repetition:
                        st.info("📌 Repetition mode: Detects when the Qari goes back and repeats previous ayahs.")

                    st.markdown("---")
                    st.selectbox(
                        "Alignment mode",
                        options=["sequential", "dtw"],
                        index=0,
                        key="alignment_mode",
                        help=(
                            "**sequential** — proven, fast, best for clean recordings. "
                            "**dtw** — globally optimal, handles stumbles, repetitions, "
                            "unusual pauses (slower)."
                        ),
                    )


                
                if st.button("🔍 Detect Ayahs", type="primary", use_container_width=True):
                    # Set logging level based on debug mode
                    if debug_mode:
                        logging.getLogger('alignment_utils').setLevel(logging.DEBUG)
                        st.info("🔧 Debug mode enabled - check terminal for detailed logs")
                    else:
                        logging.getLogger('alignment_utils').setLevel(logging.INFO)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if use_word_classification and surah_hint:
                        # Use word-level classification
                        from src.alignment_utils import reconstruct_ayahs
                        
                        status_text.text("⏳ Classifying words...")
                        progress_bar.progress(0.2)
                        
                        quran_data = load_quran_text("data/quran/quran.json")
                        normalizer = ArabicNormalizer()
                        
                        ayah_detector = AyahDetector(
                            quran_data=quran_data,
                            normalizer=normalizer,
                            confidence_threshold=confidence_threshold
                        )
                        
                        status_text.text(f"⏳ Classifying words for Surah {surah_hint}...")
                        progress_bar.progress(0.5)
                        
                        # Classify each word
                        classifications = ayah_detector.classify_transcription_words(
                            transcribed_words=ss.transcribed_words,
                            surah=surah_hint,
                            start_ayah=start_ayah,
                            end_ayah=end_ayah,
                            skip_preamble=skip_preamble
                        )
                        
                        status_text.text("⏳ Reconstructing ayahs...")
                        progress_bar.progress(0.8)
                        
                        # Reconstruct ayahs from classifications
                        detected_ayahs = reconstruct_ayahs(classifications, quran_data)
                        
                        # Store classifications for later use
                        ss.word_classifications = classifications
                        
                        progress_bar.progress(1.0)
                        status_text.text(f"✓ Classified {len(classifications)} words → {len(detected_ayahs)} ayahs")
                    else:
                        # Use standard detection workflow
                        detected_ayahs = detect_ayahs_workflow(
                            transcribed_words=ss.transcribed_words,
                            confidence_threshold=confidence_threshold,
                            surah_hint=surah_hint,
                            progress_bar=progress_bar,
                            status_text=status_text,
                            start_ayah=start_ayah,
                            end_ayah=end_ayah,
                            skip_preamble=skip_preamble,
                            allow_repetition=allow_repetition,
                            mode=ss.get("alignment_mode", "sequential"),
                        )
                    
                    if detected_ayahs:
                        ss.detected_ayahs = detected_ayahs
                        ss.excluded_event_indices = set()   # reset on new detection
                        ss.current_step = 'align'
                        
                        st.success(f"✅ Detected {len(detected_ayahs)} ayah segments")
                        st.info("👉 Proceed to the **Review & Align** tab")

                
                # Display detected ayahs if available
                if ss.get('detected_ayahs'):
                    st.divider()
                    st.subheader("🎯 Alignment Results")

                    total_dur = 0.0
                    if ss.get("transcribed_words"):
                        total_dur = ss.transcribed_words[-1].end

                    render_alignment_timeline(ss.detected_ayahs, total_dur)
                    st.markdown("---")
                    render_event_cards(ss.detected_ayahs)

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    full_n  = sum(1 for d in ss.detected_ayahs if d.get("event_type") == "full")
                    rep_n   = sum(1 for d in ss.detected_ayahs if d.get("event_type") == "repetition")
                    part_n  = sum(1 for d in ss.detected_ayahs if d.get("event_type") == "partial")
                    avg_c   = (sum(d["confidence"] for d in ss.detected_ayahs)
                               / len(ss.detected_ayahs))
                    col1.metric("Full matches", full_n)
                    col2.metric("Repetitions",  rep_n)
                    col3.metric("Partial",       part_n)
                    col4.metric("Avg confidence", f"{avg_c:.0%}")

                    low_conf = [d for d in ss.detected_ayahs
                                if d["confidence"] < confidence_threshold]
                    if low_conf:
                        st.warning(
                            f"⚠️ {len(low_conf)} ayahs below confidence threshold "
                            f"({confidence_threshold:.0%}) — review before aligning."
                        )
            
            else:
                st.warning("⚠️ Please complete transcription first")
                st.info("👈 Go back to the **Transcribe** tab")
        
        # ====================================================================
        # SUB-TAB 4: Review & Align
        # ====================================================================
        with subtab4:
            st.subheader("Review & Align Words")
            
            if ss.get('detected_ayahs'):
                st.success(f"✓ {len(ss.detected_ayahs)} ayahs detected and ready for alignment")
                
                if st.button("🔗 Align All Words", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    quran_data = load_quran_text("data/quran/quran.json")

                    _excluded = ss.get("excluded_event_indices", set())
                    _ayahs_to_align = [
                        a for i, a in enumerate(ss.detected_ayahs)
                        if i not in _excluded
                    ]
                    aligned_ayahs = align_words_workflow(
                        detected_ayahs=_ayahs_to_align,
                        transcribed_words=ss.transcribed_words,
                        quran_data=quran_data,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    
                    if aligned_ayahs:
                        ss.aligned_ayahs = aligned_ayahs
                        ss.current_step = 'export'
                        st.success(f"✅ Aligned {len(aligned_ayahs)} ayahs")
                        st.info("👉 Proceed to the **Export** tab")
                
                # Display side-by-side comparison if aligned
                if ss.get('aligned_ayahs'):
                    st.divider()
                    st.subheader("📊 Word-by-Word Alignment Review")
                    
                    # Ayah selector
                    surah_names = get_surah_names()
                    ayah_options = [
                        f"{i+1}. Surah {a['surah']}:{a['ayah']} - {surah_names.get(a['surah'], '')} ({a['confidence']:.0%} conf)"
                        for i, a in enumerate(ss.aligned_ayahs)
                    ]
                    
                    selected_idx = st.selectbox("Select Ayah to Review", range(len(ayah_options)), format_func=lambda i: ayah_options[i])
                    
                    if selected_idx is not None:
                        ayah_info = ss.aligned_ayahs[selected_idx]
                        
                        # Summary stats with normalized comparison
                        from src.audio_processing_utils import ArabicNormalizer
                        normalizer = ArabicNormalizer()
                        
                        total_words = len(ayah_info['word_alignments'])
                        ref_words = ayah_info.get('reference_words', [])
                        
                        # Count matches using normalized comparison (same as display below)
                        matched = 0
                        for tw, rp in ayah_info['word_alignments']:
                            if rp <= len(ref_words):
                                trans_normalized = normalizer.normalize(tw.word.strip())
                                ref_normalized = normalizer.normalize(ref_words[rp-1].strip())
                                if trans_normalized == ref_normalized:
                                    matched += 1
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Total Words", total_words)
                        with col_stat2:
                            st.metric("Exact Matches", f"{matched}/{total_words}")
                        with col_stat3:
                            match_rate = (matched / total_words * 100) if total_words > 0 else 0
                            st.metric("Match Rate", f"{match_rate:.0f}%")
                        
                        # Full text comparison
                        st.divider()
                        col_review1, col_review2 = st.columns(2)
                        
                        with col_review1:
                            st.markdown("**📝 Transcribed Text:**")
                            st.text_area(
                                "Transcription",
                                ayah_info['transcribed_text'],
                                height=100,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        with col_review2:
                            st.markdown("**📖 Reference Quran Text:**")
                            st.text_area(
                                "Reference",
                                " ".join(ref_words),
                                height=100,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        # Detailed word-by-word alignment
                        st.divider()
                        st.markdown("**� Detailed Word Matches:**")
                        
                        # Filter options
                        col_filter1, col_filter2 = st.columns(2)
                        with col_filter1:
                            show_filter = st.radio(
                                "Show:",
                                ["All Words", "Exact Matches Only", "Mismatches Only"],
                                horizontal=True
                            )
                        with col_filter2:
                            words_per_page = st.selectbox("Words per page:", [10, 20, 50, 100], index=1)
                        
                        # Build alignment data (reuse normalizer from stats above)
                        alignment_data = []
                        for trans_word, ref_pos in ayah_info['word_alignments']:
                            ref_word = ref_words[ref_pos - 1] if ref_pos <= len(ref_words) else "⚠️ MISSING"
                            
                            # Normalize both for comparison (ignore diacritics)
                            trans_normalized = normalizer.normalize(trans_word.word.strip())
                            ref_normalized = normalizer.normalize(ref_word.strip()) if ref_word != "⚠️ MISSING" else ""
                            is_match = trans_normalized == ref_normalized if ref_word != "⚠️ MISSING" else False
                            
                            # Apply filter
                            if show_filter == "Exact Matches Only" and not is_match:
                                continue
                            elif show_filter == "Mismatches Only" and is_match:
                                continue
                            
                            # Match indicator
                            if is_match:
                                match_icon = "✅"
                                match_status = "Match"
                            elif ref_word == "⚠️ MISSING":
                                match_icon = "⚠️"
                                match_status = "Missing"
                            else:
                                match_icon = "❌"
                                match_status = "Mismatch"
                            
                            alignment_data.append({
                                "": match_icon,
                                "Pos": ref_pos,
                                "Transcribed": trans_word.word,
                                "→": "→",
                                "Reference": ref_word,
                                "Status": match_status,
                                "Time": f"{trans_word.start:.2f}s - {trans_word.end:.2f}s",
                                "Duration": f"{trans_word.end - trans_word.start:.2f}s"
                            })
                        
                        # Pagination
                        total_filtered = len(alignment_data)
                        if total_filtered == 0:
                            st.info("ℹ️ No words match the selected filter")
                        else:
                            total_pages = (total_filtered + words_per_page - 1) // words_per_page
                            
                            col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
                            with col_page2:
                                page = st.selectbox(
                                    "Page",
                                    range(1, total_pages + 1),
                                    format_func=lambda p: f"Page {p} of {total_pages}"
                                ) - 1
                            
                            # Display paginated data
                            start_idx = page * words_per_page
                            end_idx = min(start_idx + words_per_page, total_filtered)
                            
                            st.dataframe(
                                alignment_data[start_idx:end_idx],
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_filtered} words")
                        
                        # Feedback section
                        st.divider()
                        st.markdown("**💬 Alignment Quality Feedback:**")
                        
                        col_feedback1, col_feedback2 = st.columns([3, 1])
                        with col_feedback1:
                            feedback = st.text_area(
                                "Notes on this ayah's alignment",
                                placeholder="e.g., Words 5-8 are misaligned due to reciter pausing. Word 12 missing from transcription.",
                                height=80,
                                key=f"feedback_{selected_idx}"
                            )
                        with col_feedback2:
                            quality = st.radio(
                                "Overall Quality:",
                                ["✅ Good", "⚠️ Fair", "❌ Poor"],
                                key=f"quality_{selected_idx}"
                            )
                        
                        if feedback:
                            st.info(f"💡 Your feedback: {feedback}")
                            st.caption("Note: Feedback is for review only (not saved to export)")
            
            else:
                st.warning("⚠️ Please complete ayah detection first")
                st.info("👈 Go back to the **Detect Ayahs** tab")
        
        # ====================================================================
        # SUB-TAB 5: Export
        # ====================================================================
        with subtab5:
            st.subheader("Export Tarteel JSON")
            
            if ss.get('aligned_ayahs'):
                st.success(f"✓ {len(ss.aligned_ayahs)} ayahs ready for export")
                
                output_filename = st.text_input(
                    "Output Filename",
                    "custom_recitation.json",
                    help="Name for the Tarteel-format JSON file"
                )
                
                output_dir = st.text_input(
                    "Output Directory",
                    "temp/custom_audio",
                    help="Directory to save the JSON file"
                )
                
                if st.button("💾 Export to Tarteel Format", type="primary", use_container_width=True):
                    output_path = os.path.join(output_dir, output_filename)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    with st.spinner("Exporting..."):
                        success = export_tarteel_json(
                            aligned_ayahs=ss.aligned_ayahs,
                            transcribed_words=ss.transcribed_words,
                            output_path=output_path
                        )
                    
                    if success:
                        ss.tarteel_json = output_path
                        st.success(f"✅ Successfully exported to: `{output_path}`")
                        
                        # Provide download button
                        with open(output_path, 'r', encoding='utf-8') as f:
                            json_content = f.read()
                        
                        st.download_button(
                            label="� Download JSON File",
                            data=json_content,
                            file_name=output_filename,
                            mime="application/json",
                            use_container_width=True
                        )
                        
                        # Display summary
                        st.divider()
                        st.subheader("📊 Export Summary")
                        
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        with col_sum1:
                            st.metric("Ayahs Exported", len(ss.aligned_ayahs))
                        with col_sum2:
                            total_words = sum(len(a['word_alignments']) for a in ss.aligned_ayahs)
                            st.metric("Total Words", total_words)
                        with col_sum3:
                            file_size = Path(output_path).stat().st_size / 1024
                            st.metric("File Size", f"{file_size:.1f} KB")
                        
                        st.info("""
                        **Next Steps:**
                        1. Review the JSON file to ensure quality
                        2. Use this JSON file with the **Video Generation** tab
                        3. Or integrate it into your own workflow
                        """)
            
            else:
                st.warning("⚠️ Please complete word alignment first")
                st.info("👈 Go back to the **Review & Align** tab")
    
    # ========================================================================
    # TAB 3: About
    # ========================================================================
    with tab3:
        st.header("About Quran Video Generator")
        
        st.markdown("""
        ## Overview
        
        This application generates beautiful Quran recitation videos with:
        - 📖 Word-by-word synchronized Arabic text and English translations
        - 🎬 AI-selected background videos matching verse themes
        - 🎵 High-quality recitations from renowned reciters
        - 📱 Optimized 9:16 aspect ratio for social media
        
        ## Workflows
        
        ### 1. Pre-processed Audio (Video Generation Tab)
        Uses existing Tarteel.ai recitations with word-level timestamps:
        - ✅ Instant generation (no transcription needed)
        - ✅ High accuracy timestamps
        - ✅ Multiple professional reciters
        - ⚠️ Limited to available reciters
        
        ### 2. Custom Audio Processing (Audio Tab)
        Process any Quran recitation:
        - ✅ Use any reciter's audio
        - ✅ Whisper AI transcription
        - ✅ Fuzzy matching to Quran corpus
        - ⚠️ Requires manual review (~15-30% corrections)
        - ⚠️ Longer processing time (~5-20 min per 10-min audio)
        
        ## Test Mode
        
        **Enable Test Mode** to:
        - 🎯 Test word timing without API costs
        - 🎯 Preview text rendering and fonts
        - 🎯 Verify surah/ayah selection
        - 🎯 Use black background (no Pexels/OpenAI calls)
        
        **Disable Test Mode** when ready for production videos with AI-selected backgrounds.
        
        ## System Requirements
        
        - **macOS**: Optimized for Apple Silicon (M1/M2/M3) with MPS acceleration
        - **FFmpeg**: Required for video processing
        - **Google Chrome**: Required for text rendering
        - **Python 3.8+**: With all dependencies installed
        
        ## Islamic Compliance Notice
        
        ⚠️ **Please carefully review all generated videos before use.**
        
        Background videos are sourced from external APIs and may occasionally contain content 
        that is not Islamically compliant. While we apply filters to avoid inappropriate content, 
        manual review is recommended to ensure the final video meets Islamic guidelines and standards.
        
        ## Credits
        
        - **Quran Text**: Tanzil.net
        - **Translations**: Word-by-word English translations
        - **Audio**: Tarteel.ai (for pre-processed reciters)
        - **Background Videos**: Pexels.com
        - **AI Suggestions**: OpenAI GPT-4
        - **Transcription**: OpenAI Whisper
        
        ## Support
        
        For issues, questions, or contributions, please visit the project repository.
        
        ---
        
        Made with ❤️ for the Muslim community
        """)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
