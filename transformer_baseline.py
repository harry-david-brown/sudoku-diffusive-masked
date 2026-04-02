import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader

def load_dataset(path, n=None):
    df = pd.read_csv(path, dtype=str)
    puzzles = df['quizzes'].tolist()
    solutions = df['solutions'].tolist()
    if n:
        puzzles = puzzles[:n]
        solutions = solutions[:n]
    X = torch.tensor([[int(c) for c in p] for p in puzzles], dtype=torch.long)
    Y = torch.tensor([[int(c) for c in s] for s in solutions], dtype=torch.long)
    return X, Y

class SudokuTransformer(nn.Module):
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
    rand_tensors = torch.rand(solutions.shape)
    rand_threshold = torch.rand(solutions.shape[0], 1).clamp(min=1/81)
    should_mask = unknown_mask & (rand_tensors < rand_threshold)
    corrupted = solutions.clone()
    corrupted[should_mask] = mask_token
    return corrupted, should_mask



device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
puzzles, solutions = load_dataset('sudoku.csv', n=10)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = SudokuTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


num_epochs = 20
for epoch in range(num_epochs):
    for batch_puzzles, batch_solutions in loader:
        optimizer.zero_grad()
        corrupted, should_mask = apply_mask_noise(batch_puzzles, batch_solutions)
        output = model(corrupted)
        loss = criterion(output[should_mask], batch_solutions[should_mask])
        loss.backward()
        optimizer.step()
