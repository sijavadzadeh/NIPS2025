# visualize_embeddings.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from siamese_lfp_model import CNNEncoder, SEGMENT_LENGTH, EMBEDDING_DIM

def segment_data(data, segment_length):
    num_segments = len(data) // segment_length
    segments = np.array(np.split(data[:num_segments * segment_length], num_segments))
    np.random.shuffle(segments)
    return segments

def extract_embeddings(model, data_segments, label, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for segment in data_segments:
            x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
            embedding = model.embed(x).cpu().numpy().squeeze()
            all_embeddings.append((embedding, label))
    return all_embeddings

def visualize(embeddings_with_labels):
    from sklearn.manifold import TSNE

    embeddings = np.array([e for e, _ in embeddings_with_labels])
    labels = np.array([l for _, l in embeddings_with_labels])

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"{'GPi' if label == 0 else 'STN'}", alpha=0.7)

    plt.legend()
    plt.title("Functional Embedding Space")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNEncoder(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load("siamese_model.pt", map_location=device))

    with h5py.File("gpi_data.mat", "r") as f:
        gpi_data = np.array(f["data"]).squeeze()
    with h5py.File("stn_data.mat", "r") as f:
        stn_data = np.array(f["data"]).squeeze()

    gpi_segments = segment_data(gpi_data, SEGMENT_LENGTH)[:100]  # take first 100 segments
    stn_segments = segment_data(stn_data, SEGMENT_LENGTH)[:100]

    gpi_embeddings = extract_embeddings(model, gpi_segments, label=0, device=device)
    stn_embeddings = extract_embeddings(model, stn_segments, label=1, device=device)

    all_embeddings = gpi_embeddings + stn_embeddings
    visualize(all_embeddings)

if __name__ == "__main__":
    main()
