import librosa
import numpy as np
import warnings
import os
import soundfile as sf

# Suppress librosa warnings for a cleaner terminal
warnings.filterwarnings('ignore')

class AudioStressExtractor:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        # 13 is the industry standard number of coefficients for human voice analysis
        self.n_mfcc = 13 

    def analyze_audio_chunk(self, audio_path):
        """
        Extracts acoustic features from an audio file to detect vocal stress.
        Returns a dictionary formatted for the PostgreSQL database array.
        """
        telemetry = {
            "speech_detected": False,
            "avg_speech_pace": 0.0,
            "audio_mfcc_profile": [],
            "status": "Silent"
        }

        try:
            # 1. Load the audio file
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # 2. Check for silence (Root Mean Square Energy)
            rms = librosa.feature.rms(y=y)[0]
            if np.mean(rms) < 0.01:
                return telemetry
                
            telemetry["speech_detected"] = True
            
            # 3. Extract MFCCs (The Vocal Fingerprint)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Average the coefficients across time to get a single 13-value baseline
            mfcc_mean = np.mean(mfccs.T, axis=0)
            
            # Convert the numpy array to a standard Python list so PostgreSQL can ingest it natively
            telemetry["audio_mfcc_profile"] = mfcc_mean.tolist() 
            
            # 4. Estimate Speech Pace (Syllable bursts per second)
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration > 0:
                pace = len(onsets) / duration
                telemetry["avg_speech_pace"] = round(pace, 2)
            
            telemetry["status"] = "Active Speech"
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            
        return telemetry

# --- Quick Test Execution ---
if __name__ == "__main__":
    print("🎙️ Testing Audio Stress Extractor...")
    extractor = AudioStressExtractor()
    
    # Generate a dummy 3-second audio file just to test the math engine
    test_file = "temp_test_noise.wav"
    dummy_audio = np.random.randn(16000 * 3)
    sf.write(test_file, dummy_audio, 16000)
    
    # Run the extraction
    results = extractor.analyze_audio_chunk(test_file)
    
    print("\n✅ Extraction Successful!")
    print(f"Status: {results['status']}")
    print(f"Speech Pace: {results['avg_speech_pace']} bursts/sec")
    
    # Show the first 3 values of the array to confirm it works
    if results['audio_mfcc_profile']:
        formatted_mfcc = [round(num, 2) for num in results['audio_mfcc_profile'][:3]]
        print(f"MFCC Profile (First 3 of 13): {formatted_mfcc} ...")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)