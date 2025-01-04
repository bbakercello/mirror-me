from pydub import AudioSegment
from pydub.utils import mediainfo
import os

def trim_audio(input_path, output_path, start_ms=0, end_ms=5000, output_format="wav"):
    """
    Trim an audio file to the specified range and save it efficiently.
    
    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the trimmed audio.
        start_ms (int): Start time in milliseconds.
        end_ms (int): End time in milliseconds.
        output_format (str): Format for the output file (default is "wav").
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")

        # Ensure the end time is not greater than the file duration
        info = mediainfo(input_path)
        duration_ms = float(info["duration"]) * 1000  # Convert seconds to milliseconds
        if end_ms > duration_ms:
            raise ValueError(f"End time exceeds audio duration ({duration_ms} ms).")

        # Load only the required segment
        duration = (end_ms - start_ms) / 1000  # Convert to seconds
        audio = AudioSegment.from_file(input_path, start_second=start_ms / 1000, duration=duration)
        
        # Export the trimmed audio
        audio.export(output_path, format=output_format)
        print(f"Trimmed audio saved to {output_path}")

    except Exception as e:
        print(f"Error trimming audio: {e}")

# Example usage
trim_audio(
    "./examples/audio/dev_audio/itmightcomeback.wav",
    "./examples/audio/dev_audio/trimmed_audio.wav",
    start_ms=0,
    end_ms=5000
)
