import soundfile as sf


# Replace 'your_wav_file.wav' with the path to your WAV file
file_paths = ['files\common_voice_en_128216.wav', 'files\common_voice_en_138643.wav', 'files\common_voice_en_219102.wav', 'files\common_voice_en_556731.wav', 'files\common_voice_en_657861.wav']


for i in range(len(file_paths)):
    with sf.SoundFile(file_paths[i], 'r') as file:

        num_samples = len(file)

    print(file_paths[i][-10:-4], num_samples)

