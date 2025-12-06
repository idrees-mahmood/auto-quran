# Quran Video Generator - UI Guide

## Quick Start

### 1. Launch the UI

```bash
./launch_ui.sh
```

The application will automatically open in your browser at `http://localhost:8501`

### 2. First-Time Setup

#### Configure API Keys (Optional for Test Mode)

1. Click **Settings** in the sidebar
2. Expand **"Configure API Keys"**
3. Enter your API keys:
   - **OpenAI API Key**: Get from https://platform.openai.com/api-keys
   - **Pexels API Key**: Get from https://www.pexels.com/api/
4. Click **"Save API Keys"**

> 💡 **Tip**: You can skip this step and use **Test Mode** to generate videos with black backgrounds (no API costs!)

## Features

### 🎬 Video Generation Tab

Generate videos from pre-processed Quran recitations with word-level timestamps.

#### Workflow:

1. **Select Verses**
   - Choose Surah from dropdown (e.g., "1. Al-Fatihah")
   - Set ayah range (from/to)
   - Preview shows selected range

2. **Select Reciter**
   - Mahmoud Khalil Al-Husary (Murattal Hafs)
   - Muhammad Al-Minshawi (Murattal Hafs)

3. **Choose Generation Mode**
   - ✅ **Test Mode**: Black background, no API calls, instant preview
   - 🎬 **Production Mode**: AI-selected backgrounds, requires API keys

4. **Generate**
   - Click **"🎬 Generate Video"**
   - Progress bar shows real-time status
   - Video appears below when complete

5. **Download or Preview**
   - Watch inline in browser
   - Download MP4 file
   - View file size

#### Test Mode vs Production Mode

| Feature | Test Mode | Production Mode |
|---------|-----------|-----------------|
| Background | Black screen | AI-selected videos |
| API Costs | $0.00 | ~$0.10-0.50/video |
| Speed | Fast (~2-3 min) | Slower (~5-10 min) |
| Use Case | Testing, previews | Final videos |

### 🎤 Custom Audio Processing Tab

Process any Quran recitation audio to generate word-level timestamps using Whisper AI.

#### Complete Workflow (5 Steps):

##### **Step 1: Upload Audio**

1. Click **"Upload Audio"** sub-tab
2. Use file uploader to select audio (MP3, WAV, OGG, M4A, FLAC)
3. Click **"Validate & Process Audio"**
4. Review audio metadata:
   - Duration, channels, sample rate
   - File size
5. If needed, click **"Preprocess Audio"** to convert to optimal format (16kHz mono)

**Result**: Audio ready for transcription

##### **Step 2: Transcribe**

1. Click **"Transcribe"** sub-tab
2. Select Whisper model:
   - **tiny**: Fast, ~70% accuracy (not recommended for Arabic)
   - **base**: Balanced, ~75% accuracy ⭐ **Recommended**
   - **small**: Better accuracy, slower
   - **medium/large**: Best accuracy, very slow
3. Select device:
   - **auto**: Automatic detection ⭐ (uses MPS on Apple Silicon)
   - **cpu**: Compatible with all systems
   - **mps**: Apple Silicon (M1/M2/M3) - 2-3x faster
   - **cuda**: NVIDIA GPU
4. Click **"Start Transcription"**
5. Wait for completion (5-20 minutes depending on audio length and model)
6. Review transcribed text and word timestamps

**Result**: Arabic text with word-level timestamps

**Time**: ~5-20 minutes for 10-minute audio
- tiny model: ~5 min
- base model: ~8-10 min (CPU), ~3-4 min (MPS)
- large model: ~15-20 min (CPU), ~6-8 min (MPS)

##### **Step 3: Detect Ayahs**

1. Click **"Detect Ayahs"** sub-tab
2. Set confidence threshold (70% recommended)
3. Optionally provide surah hint if you know which surah is being recited
4. Click **"Detect Ayahs"**
5. Review detected ayahs with confidence scores:
   - 🟢 Green: High confidence (>90%)
   - 🟡 Yellow: Medium confidence (70-90%)
   - 🔴 Red: Low confidence (<70%)

**Result**: Mapped ayahs with timing information

**Notes**:
- High accuracy transcription → High confidence ayah detection
- Low confidence ayahs require manual review
- Sliding window approach handles continuous recitation

##### **Step 4: Review & Align**

1. Click **"Review & Align"** sub-tab
2. Click **"Align All Words"** to map transcribed words to Quran reference
3. Select ayah from dropdown to review
4. Compare side-by-side:
   - **Left**: Transcribed text (what Whisper heard)
   - **Right**: Reference Quran text (canonical)
5. Review word-level alignments:
   - Position, timestamps, match status
   - ✓ = exact match
   - ~ = fuzzy match

**Result**: Word-level alignment with timestamps

**What to look for**:
- Check that transcribed words match reference words
- Verify timing makes sense (no huge gaps or overlaps)
- Low confidence ayahs need extra scrutiny

##### **Step 5: Export**

1. Click **"Export"** sub-tab
2. Set output filename (default: `custom_recitation.json`)
3. Set output directory (default: `temp/custom_audio`)
4. Click **"Export to Tarteel Format"**
5. Download JSON file
6. Use this JSON file with Video Generation tab or your own tools

**Result**: Tarteel-compatible JSON ready for video generation

**JSON Structure**:
```json
{
  "1:1": {
    "surah_number": 1,
    "ayah_number": 1,
    "audio_url": "custom_recitation",
    "duration": 5000,
    "segments": [[1, 0, 500], [2, 500, 1200], ...]
  }
}
```

#### Tips for Custom Audio Processing

**💡 For Best Results**:

1. **Audio Quality**:
   - Use clear, professional recitations
   - Minimize background noise
   - Avoid music or echo
   - Mono recordings work best

2. **Model Selection**:
   - Start with **base** model
   - Upgrade to **small** or **medium** if accuracy is poor
   - Use **tiny** only for quick tests

3. **Device Selection**:
   - Always use **auto** for best performance
   - On Apple Silicon (M1/M2/M3), you'll get 2-3x speedup with MPS

4. **Confidence Threshold**:
   - 70% is a good balance
   - Lower threshold = more matches, but more false positives
   - Higher threshold = fewer matches, but higher accuracy

5. **Ayah Hints**:
   - If you know the surah, provide it as hint
   - Dramatically improves accuracy
   - Reduces false matches

**💡 Troubleshooting**:

| Problem | Solution |
|---------|----------|
| Poor transcription | Try larger Whisper model (small/medium) |
| No ayahs detected | Lower confidence threshold to 60% |
| Wrong ayahs detected | Provide surah hint |
| Slow transcription | Use smaller model (base/tiny) or enable MPS |
| Out of memory | Use smaller model or shorter audio clips |
| Misaligned words | Review and manually adjust in JSON file |

**💡 Expected Accuracy**:

- **Transcription**: 70-85% for Arabic (varies by reciter, audio quality)
- **Ayah Detection**: 80-95% (with confidence >70%)
- **Word Alignment**: 85-95% (some interpolation needed)

**💡 Manual Corrections**:

After export, you can manually edit the JSON file:
1. Open `custom_recitation.json` in text editor
2. Adjust timestamps: `[word_position, start_ms, end_ms]`
3. Add missing words
4. Remove incorrect detections
5. Save and use for video generation

#### Workflow Comparison

| Feature | Pre-processed (Tab 1) | Custom Audio (Tab 2) |
|---------|----------------------|---------------------|
| **Speed** | Instant | 5-20 minutes |
| **Accuracy** | 99%+ (manual timing) | 70-85% (AI) |
| **Reciters** | Limited (2 available) | Any reciter |
| **Cost** | Free | Free (Whisper is local) |
| **Effort** | Select & generate | Upload, review, correct |
| **Use Case** | Quick videos | Custom reciters |

### ⚙️ Settings (Sidebar)

#### API Keys
- Store OpenAI and Pexels API keys
- Persists during session
- Can be set via environment variables

#### Video Settings
- **Font**: Choose from available Arabic fonts
  - Rakkas (default, bold and decorative)
  - Aref Ruqaa (classic calligraphic)
  - Joumhouria (modern)
- **Font Size**: Adjust text size (40-150px)

#### Advanced
- **Clean Temp Files**: Remove temporary audio/video files

## Common Tasks

### Generate a Test Video (Free)

1. Keep **Test Mode** enabled ✅
2. Select Surah 1 (Al-Fatihah), ayahs 1-7
3. Choose any reciter
4. Click **Generate Video**
5. Wait ~2-3 minutes
6. Preview and download

**Result**: Black background video with synchronized Arabic text and English translations

### Generate a Production Video

1. Add API keys in Settings
2. **Disable Test Mode** ❌
3. Select your verses
4. Click **Generate Video**
5. Wait ~5-10 minutes (AI generates background suggestions and downloads videos)
6. **Important**: Review video for Islamic compliance before use!

### Change Font or Size

1. Open sidebar **Settings**
2. Select different font from dropdown
3. Adjust size slider
4. Generate new video with updated styling

### Clean Up Old Files

1. Open sidebar **Settings**
2. Click **"🗑️ Clean Temp Files"**
3. Removes all temporary audio/video files

## UI Components

### Progress Tracking

Real-time progress bar shows:
- ⏳ Loading Quran data
- ⏳ Adjusting timestamps
- ⏳ Generating AI suggestions (production mode)
- ⏳ Downloading videos (production mode)
- ⏳ Composing final video
- ✓ Complete!

### Status Messages

- 🟢 **Success**: Green boxes for completed actions
- 🟡 **Warning**: Yellow boxes for non-critical issues
- 🔴 **Error**: Red boxes for failures with details
- 🔵 **Info**: Blue boxes for helpful information

### Video Preview

- Inline HTML5 video player
- Full quality playback
- Download button for offline use
- File size indicator
- Clear button to free up space

## Keyboard Shortcuts

- **Ctrl+C** (in terminal): Stop the UI server
- **R** (in browser): Rerun the app (useful after errors)
- **C** (in browser): Clear cache

## Troubleshooting

### UI Won't Start

**Problem**: `./launch_ui.sh` fails

**Solution**:
```bash
# Check Python environment
source venv/bin/activate
pip install -r requirements.txt

# Try manual launch
streamlit run app.py
```

### Video Generation Fails

**Problem**: Error during video generation

**Solutions**:
1. Check API keys are valid (production mode)
2. Verify FFmpeg is installed: `ffmpeg -version`
3. Verify Chrome is installed (required for text rendering)
4. Check temp/ directory is writable
5. Review error details in red box

### Background Videos Don't Appear

**Problem**: Only black background shows

**Solutions**:
1. Verify **Test Mode is disabled**
2. Check API keys are entered in Settings
3. Ensure internet connection is stable
4. Check Pexels API quota hasn't been exceeded

### Text Not Rendering

**Problem**: No text overlays appear

**Solutions**:
1. Verify Google Chrome is installed
2. Check font files exist in `data/fonts/`
3. Try different font from dropdown
4. Check browser console for errors

### Slow Performance

**Problem**: Video generation takes too long

**Solutions**:
1. Use **Test Mode** for faster testing
2. Reduce ayah range (process fewer verses)
3. On macOS: Ensure MPS acceleration is enabled
4. Close other heavy applications

### API Costs Too High

**Problem**: Don't want to spend on API credits

**Solution**:
- ✅ **Use Test Mode!**
- Perfect for testing, previews, and font adjustments
- Only use production mode for final videos
- One test run → validate timing → one production run

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection (for API calls)

### Recommended
- Python 3.10+
- 8GB RAM
- 10GB free disk space
- macOS with Apple Silicon (M1/M2/M3) for acceleration
- Stable broadband connection

### Required Software
- **FFmpeg**: Video processing
- **Google Chrome**: Text rendering
- **Python packages**: Automatically installed via requirements.txt

## Tips & Best Practices

### 💡 Development Workflow

1. **First iteration**: Use Test Mode
   - Verify surah/ayah selection
   - Check text timing and synchronization
   - Test font choices and sizes
   
2. **Second iteration**: Production Mode
   - Generate final video with backgrounds
   - Review for Islamic compliance
   - Download and share

### 💡 Font Selection

- **Rakkas**: Bold, modern, great for social media
- **Aref Ruqaa**: Traditional, elegant, good for formal content
- **Joumhouria**: Clean, readable, good for educational content

### 💡 Ayah Range

- **Short clips** (1-3 ayahs): Better for social media, faster generation
- **Medium clips** (4-10 ayahs): Good for thematic content
- **Long clips** (10+ ayahs): Use for complete surahs, but slower

### 💡 API Cost Management

- Test Mode is **completely free**
- Production mode costs vary:
  - Short clip (1-3 ayahs): ~$0.05-0.15
  - Medium clip (4-10 ayahs): ~$0.15-0.40
  - Long clip (10+ ayahs): ~$0.40-1.00

### 💡 Islamic Compliance

⚠️ **Always review generated videos**

Background videos come from Pexels API. While filtered, manual review ensures:
- No inappropriate imagery
- No music or prohibited content
- Alignment with Islamic values
- Suitable for intended audience

## Keyboard Shortcuts (Streamlit)

- **R**: Rerun the entire app
- **C**: Clear cache
- **Ctrl+Shift+K**: Open command palette
- **Ctrl+K**: Focus search
- **?**: Show all shortcuts

## Browser Compatibility

✅ **Recommended**: Google Chrome, Microsoft Edge
✅ **Supported**: Firefox, Safari
⚠️ **Limited**: Older browsers (may have video playback issues)

## Getting Help

### Check Logs

Terminal output shows detailed progress and errors. Look for:
- Red error messages
- Stack traces
- API error codes

### Report Issues

When reporting issues, include:
1. Error message (full text)
2. Steps to reproduce
3. Browser and OS version
4. Screenshot (if applicable)

### Community Support

- Check existing documentation
- Review example videos in `examples/`
- Test with known-working configurations first

---

## Next Steps

1. ✅ **Try Test Mode**: Generate your first video in minutes
2. 📖 **Explore Options**: Test different fonts, surahs, reciters
3. 🎬 **Go Production**: Add API keys and generate final videos
4. 📤 **Share**: Download and share your videos

---

Made with ❤️ for the Muslim community
