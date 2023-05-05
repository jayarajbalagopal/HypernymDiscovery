import torch
import torch.nn as nn
import pickle
import sys


from models import HypernymHyponymLSTM
from models import HypernymHyponymGRU
from models import HypernymHyponymFC

with open('./1A/vocab_hypernym.pkl', 'rb') as f:
    vocab_hypernym = pickle.load(f)
    f.close()

with open('./1A/vocab_hyponym.pkl', 'rb') as f:
    vocab_hyponym = pickle.load(f)
    f.close()

with open('./1A/all_hypernyms.pkl', 'rb') as f:
    all_hypernyms = pickle.load(f)
    f.close()

idx_to_word = {}
for k, v in vocab_hypernym.items():
    idx_to_word[v]=k

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_candidate_hypernyms(model, hyponym):

    hyponym = hyponym.lower()
        
    try:
        # Encode the hyponym using the vocabulary
        hyponym_idx = vocab_hyponym[hyponym]
    except(KeyError):
        print('Please provide a valid hyponym')
        sys.exit()

    # Convert the hyponym index to a tensor and move to device
    hyponym_tensor = torch.LongTensor([hyponym_idx]).to(device)

    # Predict the hypernym logits using the trained model
    hypernym_logits = model(hyponym_tensor)

    softmax = nn.Softmax(dim=1)
    # Convert the logits to probabilities using softmax
    hypernym_probs = softmax(hypernym_logits)

    # Extract the top 10 predicted hypernyms and their probabilities
    top_k_probs, top_k_indices = torch.topk(hypernym_probs, k=10)
    top_k_hypernyms = [idx_to_word[idx.item()] for idx in top_k_indices[0]]

    # Print the results
    print(f"Top 10 predicted hypernyms for '{hyponym}':")
    for hypernym, prob in zip(top_k_hypernyms, top_k_probs[0]):
        print(f"{hypernym}: {prob.item():.4f}")


#UNCOMMENT THE MODEL AS PER THE USE

model = HypernymHyponymLSTM(num_hyponyms=len(vocab_hyponym), num_hypernyms=len(all_hypernyms), embedding_dim=200, hidden_dim=400).to(device)
# model = HypernymHyponymGRU(num_hyponyms=len(vocab_hyponym), num_hypernyms=len(all_hypernyms), embedding_dim=200, hidden_dim=400).to(device)
# model = HypernymHyponymFC(num_hyponyms=len(vocab_hyponym), num_hypernyms=len(all_hypernyms), embedding_dim=200, hidden_dim=400).to(device)

# Define the path to the .pt file (change accordingly)
path = "./1A/model_LSTM.pt"

# Load the model from the file and load it onto the CPU
model = torch.load(path, map_location=torch.device('cpu'))

print('Model loaded successfully')

# Set the model to evaluation mode
model.eval()

hyponym = input("Please input the hyponym, ")

predict_candidate_hypernyms(model, hyponym)