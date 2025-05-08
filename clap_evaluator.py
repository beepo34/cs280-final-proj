'''Script for evaluating audio .npy files with CLAP scores. You have to download the audio files from colab to avoid dependency issues.
To run this script, download audio files from colab and edit text_prompts and audio_labels variables to match audio inputs.'''
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import torch

# Install Laion-clap
try:
    from laion_clap import CLAP_Module
except ImportError:
    print("Installing laion_clap...")
    import subprocess
    subprocess.check_call(["pip", "install", "git+https://github.com/LAION-AI/CLAP.git"])
    from laion_clap import CLAP_Module

def load_audio_files(file_paths):
    audio_data = []
    for file_path in file_paths:
        try:
            data = np.load(file_path)
            audio_data.append(data)
            print(f"Successfully loaded {file_path}, shape: {data.shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return audio_data

def save_as_wav(audio_data, file_paths, sample_rate=16000):
    wav_paths = []
    for data, file_path in zip(audio_data, file_paths):
        wav_path = file_path.replace('.npy', '.wav')
        flattened_data = np.squeeze(data)
        normalized_data = flattened_data / (np.max(np.abs(flattened_data)) + 1e-8)
        try:
            sf.write(wav_path, normalized_data, sample_rate)
            wav_paths.append(wav_path)
            print(f"Saved WAV file: {wav_path}")
        except Exception as e:
            print(f"Error saving {wav_path}: {e}")
    return wav_paths

def evaluate_with_clap(wav_paths, text_prompts, use_cuda=False):
    print("Initializing LAION-CLAP model...")
    clap_model = CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()  # Load pre-trained checkpoint

    if use_cuda:
        clap_model.cuda()

    text_embeddings = clap_model.get_text_embedding(text_prompts)
    audio_embeddings = clap_model.get_audio_embedding_from_filelist(wav_paths)
    
    text_embeddings = torch.tensor(text_embeddings)
    audio_embeddings = torch.tensor(audio_embeddings)

    # Normalize the embeddings before similarity
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    audio_embeddings = audio_embeddings / audio_embeddings.norm(dim=-1, keepdim=True)
    similarities = (audio_embeddings @ text_embeddings.T).detach().cpu().numpy()
    
    return similarities

def visualize_results(similarities, text_prompts, audio_labels):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(text_prompts))
    width = 0.4

    for i, label in enumerate(audio_labels):
        plt.bar(x + i * width, similarities[i], width, label=label)

    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.xlabel('Text Prompts')
    plt.ylabel('Similarity Score')
    plt.title('CLAP Audio-Text Similarity')
    plt.xticks(x + width / 2, text_prompts, rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('clap_barplot.png')
    plt.show()

    # Heatmap
    plt.figure(figsize=(8, 5))
    plt.imshow(similarities, cmap='viridis')
    plt.colorbar(label='Similarity Score')
    plt.xticks(np.arange(len(text_prompts)), text_prompts, rotation=45, ha='right')
    plt.yticks(np.arange(len(audio_labels)), audio_labels)
    plt.title('CLAP Similarity Heatmap')
    for i in range(len(audio_labels)):
        for j in range(len(text_prompts)):
            plt.text(j, i, f"{similarities[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if similarities[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig('clap_heatmap.png')
    plt.show()

def main():
    npy_files = ['forward_audio.npy', 'flipped_audio.npy']
    audio_labels = ['Forward (dog)', 'Flipped (cat)']
    text_prompts = ['a dog whining', 'a cat meowing']
    use_cuda = False

    print("Loading audio data...")
    audio_data = load_audio_files(npy_files)
    if not audio_data:
        print("No audio data loaded. Exiting.")
        return

    print("Saving as WAV...")
    wav_paths = save_as_wav(audio_data, npy_files)
    if not wav_paths:
        print("No WAV files created. Exiting.")
        return

    similarities = evaluate_with_clap(wav_paths, text_prompts, use_cuda)

    with open("clap_evaluation.txt", "w") as f:
        f.write("CLAP Similarity Scores:\n")
        for i, label in enumerate(audio_labels):
            f.write(f"\n{label}:\n")
            for j, prompt in enumerate(text_prompts):
                f.write(f"  {prompt}: {similarities[i][j]:.4f}\n")


    visualize_results(similarities, text_prompts, audio_labels)

    with open("clap_evaluation.txt", "a") as f:  # append mode to continue writing
        f.write("\nAnalysis of the Results:\n")
        for i, label in enumerate(audio_labels):
            best_match = np.argmax(similarities[i])
            f.write(f"{label} matches best with '{text_prompts[best_match]}' (score: {similarities[i][best_match]:.4f})\n")

if __name__ == "__main__":
    main()
