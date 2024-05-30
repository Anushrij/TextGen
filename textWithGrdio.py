import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
import gradio as gr
from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# #
# df = pd.read_parquet('oz1.parquet')
# df.to_csv('oz1.csv')


# Step 1: Load and Preprocess the Large Text Dataset

with open('oz.txt', 'r', encoding='utf-8') as file:
    text = file.read().replace('\n', ' ')

# Preprocessing the text data
chars = list(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
data = [char_to_idx[c] for c in text]  # Convert characters to indices

# Hyperparameters
seq_length = 10
input_size = len(chars)
hidden_size = 128
num_layers = 2
learning_rate = 0.001
batch_size = 64

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Define the dataset
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_length]
        target = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(seq), torch.tensor(target)

def generate_text(model, start_text, length):
    model.eval()
    hidden = model.init_hidden(1)
    input_seq = [char_to_idx[ch] for ch in start_text]
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    generated_text = start_text

    for _ in range(length):
        input_seq = nn.functional.one_hot(input_seq, num_classes=input_size).float()
        output, hidden = model(input_seq, hidden)
        output_char = idx_to_char[torch.argmax(output[-1]).item()]
        generated_text += output_char
        input_seq = torch.tensor([[char_to_idx[output_char]]]).to(device)

    return generated_text

# Main training function
def main():
    # Initialize dataset and dataloader
    dataset = TextDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers= 0)  # Set num_workers to 0 for Windows compatibility

    # Initialize model, loss function, and optimizer
    global model  # Ensure model is globally accessible
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for seq, target in dataloader:
            hidden = model.init_hidden(seq.size(0))  # Initialize hidden state with the current batch size
            seq = nn.functional.one_hot(seq, num_classes=input_size).float().to(device)
            target = target.to(device)
            hidden = tuple([each.data for each in hidden])  # Detach hidden states to prevent backprop through the entire history

            output, hidden = model(seq, hidden)

            # Ensure shapes are compatible with CrossEntropyLoss
            output = output.view(-1, input_size)
            target = target.view(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Generate text
    start_text = "hello"
    generated_text = generate_text(model, start_text, 100)
    print(generated_text)

# Ensure the main function is called only when the script is run directly
if __name__ == '__main__':
    main()

# Gradio interface
def gradio_interface(start_text, length):
    return generate_text(model, start_text, length)

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(lines=2, placeholder="Enter start text here..."), gr.Slider(10, 500, step=10)],
    outputs=gr.Textbox()
)

interface.launch()

