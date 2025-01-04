from pydub import AudioSegment

def trim_audio(input_path, output_path, start_ms=0, end_ms=5000):
    """
    Trim an audio file to the specified range and save it.
    
    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the trimmed audio.
        start_ms (int): Start time in milliseconds.
        end_ms (int): End time in milliseconds.
    """
    audio = AudioSegment.from_file(input_path)
    trimmed_audio = audio[start_ms:end_ms]
    trimmed_audio.export(output_path, format="wav")
    print(f"Trimmed audio saved to {output_path}")

# Example usage
trim_audio("./examples/audio/dev_audio/itmightcomeback.wav", "./examples/audio/dev_audio/trimmed_audio.wav", start_ms=0, end_ms=5000)
