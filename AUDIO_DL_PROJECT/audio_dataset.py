import torch
import torchaudio
import librosa

genres = ['Hip-Hop', 'Rock', 'Pop', 'Folk', 'Experimental', 'Electronic', 'Instrumental', 'International']

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, genre_labels, feature_extractor, max_length):
        self.audio_files = audio_files
        self.genre_labels = genre_labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.audio_files[idx], sr=16000, duration=self.max_length)
        audio = torch.from_numpy(audio).float()

        features = self.feature_extractor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = features.input_values.squeeze(0) # Remove extra dimensions from input_values

        label = torch.tensor(genres.index(self.genre_labels[idx]))
        return input_values[:479626], label