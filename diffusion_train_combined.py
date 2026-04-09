import torch
import torch.nn as nn
import pandas as pd
import time
import random
from torch.utils.data import TensorDataset, DataLoader

EASY_DATASET_PATH = 'sudoku.csv'
HARD_DATASET_PATH = '/Users/harry/.cache/huggingface/hub/datasets--imone--sudoku-hard-v2/snapshots/58942f96baeb572ca3127e2a9e9c70f330783d6b/train.csv'


def load_combined_dataset(easy_path, hard_path, n_easy=250000, n_hard=250000, min_rating=50):
    # load easy
    easy_df = pd.read_csv(easy_path, dtype=str)
    easy_puzzles   = easy_df['quizzes'].tolist()[:n_easy]
    easy_solutions = easy_df['solutions'].tolist()[:n_easy]
    print(f"Easy puzzles loaded: {len(easy_puzzles):,}")

    # load hard
    hard_df = pd.read_csv(hard_path, dtype=str)
    hard_df['rating'] = pd.to_numeric(hard_df['rating'], errors='coerce')
    hard_df = hard_df[hard_df['rating'] > min_rating].iloc[:n_hard]
    hard_puzzles   = [p.replace('.', '0') for p in hard_df['question'].tolist()]
    hard_solutions = [s.replace('.', '0') for s in hard_df['answer'].tolist()]
    print(f"Hard puzzles loaded: {len(hard_puzzles):,} (rating > {min_rating})")

    # combine and shuffle
    combined = list(zip(easy_puzzles + hard_puzzles, easy_solutions + hard_solutions))
    random.shuffle(combined)
    all_puzzles, all_solutions = zip(*combined)
    print(f"Total combined: {len(all_puzzles):,}")

    X = torch.tensor([[int(c) for c in p] for p in all_puzzles], dtype=torch.long)
    Y = torch.tensor([[int(c) for c in s] for s in all_solutions], dtype=torch.long)
    return X, Y


class SudokuDiffusion(nn.Module):
    def __init__(self, vocab_size=11, embed_dim=128, num_heads=4, num_layers=4, seq_len=81):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, 10)
        self.seq_len = seq_len

    def forward(self, x):
        positions = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        return self.output(x)


def apply_mask_noise(puzzles, solutions, mask_token=10):
    unknown_mask = (puzzles == 0)
    rand_tensors = torch.rand(solutions.shape, device=solutions.device)
    rand_threshold = torch.rand(solutions.shape[0], 1, device=solutions.device).clamp(min=1/81)
    should_mask = unknown_mask & (rand_tensors < rand_threshold)
    corrupted = solutions.clone()
    corrupted[should_mask] = mask_token
    return corrupted, should_mask


# ── Setup ──────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

puzzles, solutions = load_combined_dataset(EASY_DATASET_PATH, HARD_DATASET_PATH)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SudokuDiffusion().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training ───────────────────────────────────────────────────────────────────
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start = time.time()
    for batch_puzzles, batch_solutions in loader:
        batch_puzzles = batch_puzzles.to(device)
        batch_solutions = batch_solutions.to(device)
        optimizer.zero_grad()
        corrupted, should_mask = apply_mask_noise(batch_puzzles, batch_solutions)
        output = model(corrupted)
        loss = criterion(output[should_mask], batch_solutions[should_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {total_loss/len(loader):.4f} — {elapsed:.0f}s")

torch.save(model.state_dict(), 'sudoku_diffusion_masked_combined.pth')
print("Model saved.")
