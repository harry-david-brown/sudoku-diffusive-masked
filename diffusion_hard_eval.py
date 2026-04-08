import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader

HARD_DATASET_PATH = '/Users/harry/.cache/huggingface/hub/datasets--imone--sudoku-hard-v2/snapshots/58942f96baeb572ca3127e2a9e9c70f330783d6b/train.csv'

def load_hard_dataset(path, n=None, min_rating=50):
    df = pd.read_csv(path, dtype=str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'] > min_rating].copy()
    print(f"Hard puzzles available (rating > {min_rating}): {len(df):,}")
    if n:
        df = df.iloc[:n]
    puzzles   = df['question'].tolist()
    solutions = df['answer'].tolist()
    X = torch.tensor([[0 if c == '.' else int(c) for c in p] for p in puzzles], dtype=torch.long)
    Y = torch.tensor([[0 if c == '.' else int(c) for c in s] for s in solutions], dtype=torch.long)
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

puzzles, solutions = load_hard_dataset(HARD_DATASET_PATH, n=500000)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SudokuDiffusion().to(device)
model.load_state_dict(torch.load('sudoku_diffusion_masked_500k.pth', map_location=device))
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lower LR for fine-tuning

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

torch.save(model.state_dict(), 'sudoku_diffusion_masked_finetuned_hard.pth')
print("Model saved.")