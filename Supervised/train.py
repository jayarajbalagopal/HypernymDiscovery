
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# UNCOMMENT ANY ONE OF THE TASK FILES and REPLACE THE READLINES FUNCTION 

#ENGLISH
file_hypernym_1A_train = open('./SemEval2018-Task9/training/gold/1A.english.training.gold.txt')
file_hyponym_1A_train = open('./SemEval2018-Task9/training/data/1A.english.training.data.txt')

file_hypernym_1A_test = open('./SemEval2018-Task9/test/gold/1A.english.test.gold.txt')
file_hyponym_1A_test = open('./SemEval2018-Task9/test/data/1A.english.test.data.txt')


#ITALIAN
# file_hypernym_1B_train = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/training/gold/1B.italian.training.gold.txt')
# file_hyponym_1B_train = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/training/data/1B.italian.training.data.txt')

# file_hypernym_1B_test = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/test/gold/1B.italian.test.gold.txt')
# file_hyponym_1B_test = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/test/data/1B.italian.test.data.txt')

#SPANISH
# file_hypernym_1C_train = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/training/gold/1C.spanish.training.gold.txt')
# file_hyponym_1C_train = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/training/data/1C.spanish.training.data.txt')

# file_hypernym_1C_test = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/test/gold/1C.spanish.test.gold.txt')
# file_hyponym_1C_test = open('/content/drive/MyDrive/NLP_Project/SemEval2018-Task9/test/data/1C.spanish.test.data.txt')

train_hypernym = file_hypernym_1A_train.readlines()
train_hyponym = file_hyponym_1A_train.readlines()

test_hypernym = file_hypernym_1A_test.readlines()
test_hyponym = file_hyponym_1A_test.readlines()



def create_dictionary(vocab):
    word_to_idx, idx_to_word = dict(), dict()
    for i, word in enumerate(vocab):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return word_to_idx, idx_to_word

def remove_spaces(data):
    result = []
    for i in range(len(data)):
        words = data[i].strip().split('\t')
        final_word = ""
        for word in words:
            indiv_words = word.split(" ")
            
            joined_word = ("_").join(indiv_words).lower()
            final_word += joined_word+'\t'
        result.append(final_word)
    return result





def preprocess(hypernym, hyponym):
    preprocessed_hyponym = remove_spaces(hyponym)
    preprocessed_hypernym = remove_spaces(hypernym)

    # pre_processed_vocab = remove_spaces(vocab)

    hyponyms_final, hypernyms_final = [], []

    for i in range(len(preprocessed_hypernym)):
        hypernyms_final.append(preprocessed_hypernym[i].strip().split('\t'))
    
    for i in range(len(preprocessed_hyponym)):
        hyponyms_final.append(preprocessed_hyponym[i].strip().split('\t')[0])

    # word_to_idx, idx_to_word = create_dictionary(pre_processed_vocab)

    return hyponyms_final, hypernyms_final

preprocessed_hyponyms, preprocessed_hypernyms = preprocess(train_hypernym, train_hyponym)
test_hyponyms, test_hypernyms = preprocess(train_hypernym, test_hyponym)

all_hypernyms = list(set(item for hypernyms in preprocessed_hypernyms for item in hypernyms))

def negative_samples(idx, num_of_neg=100):
    negatives = []
    for i in range(num_of_neg):
        
        rand_idx = np.random.randint(len(all_hypernyms))
        if(all_hypernyms[rand_idx] not in preprocessed_hypernyms[idx]):
            negatives.append(all_hypernyms[rand_idx])
    
    return negatives

negative_hypernyms_final = []
for idx in range(len(preprocessed_hyponyms)):
    negative_hypernyms_final.append(negative_samples(idx))

vocab_hypernym = {}
vocab_hyponym = {}
index_hypernym = 0
index_hyponym = 0
for hypernyms in all_hypernyms:
    if hypernyms not in vocab_hypernym:
        vocab_hypernym[hypernyms] = index_hypernym
        index_hypernym += 1

for hyponyms in preprocessed_hyponyms:
    if hyponyms not in vocab_hyponym:
        vocab_hyponym[hyponyms] = index_hyponym
        index_hyponym += 1

for hyponyms in test_hyponyms:
    if hyponyms not in vocab_hyponym:
        vocab_hyponym[hyponyms] = index_hyponym
        index_hyponym += 1

# len(preprocessed_hyponyms)

hypernyms = []
for hypernym in preprocessed_hypernyms:
    res = []
    for item in hypernym:
        res.append(vocab_hypernym[item])
    hypernyms.append(res)
hyponyms = [vocab_hyponym[h] for h in preprocessed_hyponyms]
negative_hypernyms = []
for n_hypernym in negative_hypernyms_final:
    res = []
    for item in n_hypernym:
        res.append(vocab_hypernym[item])
    negative_hypernyms.append(res)



"""## Architecture"""


class HypernymHyponymLSTM(nn.Module):
    def __init__(self, num_hyponyms, num_hypernyms, embedding_dim, hidden_dim):
        super(HypernymHyponymLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hyponym_embedding = nn.Embedding(num_hyponyms, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_hypernyms)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hyponym):
        hyponym_embedded = self.hyponym_embedding(hyponym)
        lstm_out1, _ = self.lstm1(hyponym_embedded)
        lstm_out2, _ = self.lstm2(lstm_out1)
        linear_out = self.linear(lstm_out2)
        return linear_out

class HypernymHyponymGRU(nn.Module):
    def __init__(self, num_hyponyms, num_hypernyms, embedding_dim, hidden_dim):
        super(HypernymHyponymGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hyponym_embedding = nn.Embedding(num_hyponyms, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, num_layers=2)  # Two GRU layers
        self.gru2 = nn.GRU(hidden_dim, hidden_dim)  # Second GRU layer
        self.linear = nn.Linear(hidden_dim, num_hypernyms)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hyponym):
        hyponym_embedded = self.hyponym_embedding(hyponym)
        gru1_out, _ = self.gru1(hyponym_embedded)
        gru2_out, _ = self.gru2(gru1_out)  # Pass output of first GRU layer to second GRU layer
        linear_out = self.linear(gru2_out)
        return linear_out

class HypernymHyponymFC(nn.Module):
    def __init__(self, num_hyponyms, num_hypernyms, embedding_dim, hidden_dim):
        super(HypernymHyponymFC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hyponym_embedding = nn.Embedding(num_hyponyms, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_hypernyms)
        self.dropout = nn.Dropout(0.5)

    def forward(self, hyponym):
        hyponym_embedded = self.hyponym_embedding(hyponym)
        linear1_out = self.dropout(self.linear1(hyponym_embedded))
        linear2_out = self.linear2(linear1_out)
        return linear2_out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelLSTM = HypernymHyponymLSTM(num_hyponyms=len(vocab_hyponym), num_hypernyms=len(all_hypernyms), embedding_dim=200, hidden_dim=400).to(device)
modelGRU = HypernymHyponymGRU(num_hyponyms=len(vocab_hyponym), num_hypernyms=len(all_hypernyms), embedding_dim=200, hidden_dim=400).to(device)
optimizerLSTM = optim.Adam(modelLSTM.parameters(), lr=0.0005)
optimizerGRU = optim.Adam(modelGRU.parameters(), lr=0.0005)

modelFC = HypernymHyponymFC(num_hyponyms=len(vocab_hyponym), num_hypernyms=len(all_hypernyms), embedding_dim=200, hidden_dim=400).to(device)
optimizerFC = optim.Adam(modelFC.parameters(),lr=0.0005)

# Set up the loss function for predicting hypernyms
hypernym_loss_fn = nn.CrossEntropyLoss()

hyponym_hypernym_dict = {}
hyponym_negative_hypernym_dict = {}
for idx in range(len(hyponyms)):
    hyponym_hypernym_dict[hyponyms[idx]] = hypernyms[idx]

    hyponym_negative_hypernym_dict[hyponyms[idx]] = negative_hypernyms[idx]



def train(model, optimizer, hyponym_hypernym_dict, hyponym_negative_hypernym_dict, num_epochs, batch_size=32):
    # Create empty lists to store the loss values for each epoch
    positive_losses = []
    negative_losses = []

    for epoch in range(num_epochs):
        print("Starting epoch", epoch+1)
        model.train()
        total_positive_loss = 0
        total_negative_loss = 0

        
        
        # Convert the hyponym_hypernym_dict and hyponym_negative_hypernym_dict into lists of tuples
        hyponym_hypernym_pairs = [(hyponym, hypernym) for hyponym, hypernyms in hyponym_hypernym_dict.items() for hypernym in hypernyms]
        hyponym_negative_hypernym_pairs = [(hyponym, negative_hypernym) for hyponym, negative_hypernyms in hyponym_negative_hypernym_dict.items() for negative_hypernym in negative_hypernyms]
        
        # Shuffle the pairs to introduce randomness in batch creation
        random.shuffle(hyponym_hypernym_pairs)
        random.shuffle(hyponym_negative_hypernym_pairs)
        
        # Calculate the total number of mini-batches for positive and negative samples
        num_positive_batches = len(hyponym_hypernym_pairs) // batch_size
        num_negative_batches = len(hyponym_negative_hypernym_pairs) // batch_size

        # Process positive samples in mini-batches
        for i in range(num_positive_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_pairs = hyponym_hypernym_pairs[start_idx:end_idx]
            optimizer.zero_grad()

            hyponyms = torch.LongTensor([pair[0] for pair in batch_pairs]).to(device)
            hypernyms = torch.LongTensor([pair[1] for pair in batch_pairs]).to(device)
            predictions = model(hyponyms)
            
            positive_loss = hypernym_loss_fn(predictions, hypernyms)
            
            positive_loss.backward()
            optimizer.step()
            total_positive_loss += positive_loss.item()

        # Process negative samples in mini-batches
        for i in range(num_negative_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_pairs = hyponym_negative_hypernym_pairs[start_idx:end_idx]
            optimizer.zero_grad()

            hyponyms = torch.LongTensor([pair[0] for pair in batch_pairs]).to(device)
            negative_hypernyms = torch.LongTensor([pair[1] for pair in batch_pairs]).to(device)
            predictions = model(hyponyms)
            
            # Calculate the negative loss using margin loss
            margin = 0.1
            negative_scores = predictions.gather(1, negative_hypernyms.view(-1,1)).squeeze()
            positive_scores = predictions.gather(1, hyponyms.view(-1,1)).squeeze()
            negative_loss = torch.mean(torch.max(torch.zeros_like(negative_scores), margin - negative_scores + positive_scores))
            
            negative_loss.backward()
            optimizer.step()
            total_negative_loss += negative_loss.item()

        avg_positive_loss = total_positive_loss / num_positive_batches
        avg_negative_loss = total_negative_loss / num_negative_batches

        positive_losses.append(avg_positive_loss)
        negative_losses.append(avg_negative_loss)

        print("Epoch:", epoch+1, "Positive Loss:", avg_positive_loss, "Negative Loss:", avg_negative_loss)

    

    plt.figure()
    plt.plot(range(1, num_epochs+1), positive_losses, label='Positive Loss')
    plt.title("Positive Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(1, num_epochs+1), negative_losses, label='Negative Loss')
    plt.title("Negative Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train the model and save the model
train(modelGRU, optimizerGRU, hyponym_hypernym_dict, hyponym_negative_hypernym_dict, num_epochs=15)
torch.save(modelGRU, 'model_GRU.pt')

train(modelLSTM, optimizerLSTM, hyponym_hypernym_dict, hyponym_negative_hypernym_dict, num_epochs=15)
torch.save(modelLSTM, 'model_LSTM.pt')

train(modelFC, optimizerFC, hyponym_hypernym_dict, hyponym_negative_hypernym_dict, num_epochs=15)
torch.save(modelFC, 'model_FC.pt')

def calculate_MAP(model, test_hyponyms, test_hypernyms, k=10):
    model.eval()
    APs = []

    for i in range(len(test_hyponyms)):
        hyponym = test_hyponyms[i]
        hyponym_idx = vocab_hyponym[hyponym]

        # Convert the hyponym index to a tensor and move to device
        hyponym_tensor = torch.LongTensor([hyponym_idx]).to(device)

        # Predict the hypernym logits using the trained model
        hypernym_logits = model(hyponym_tensor)

        softmax = nn.Softmax(dim=1)
        # Convert the logits to probabilities using softmax
        hypernym_probs = softmax(hypernym_logits)

        # Extract the top 10 predicted hypernyms and their probabilities
        top_k_probs, top_k_indices = torch.topk(hypernym_probs, k=len(test_hypernyms[i]))
        top_k_hypernyms = [idx_to_word[idx.item()] for idx in top_k_indices[0]]

        AP = calculate_AP(top_k_hypernyms, test_hypernyms[i])
        APs.append(AP)

    return np.mean(APs)

def calculate_AP(predicted_hypernyms, labels):
    num_correct = 0
    total_precision = 0
    for i, hypernym in enumerate(predicted_hypernyms):
        if(i==len(labels)):
            break
        if hypernym in labels:
            num_correct += 1
            precision = num_correct / (i+1)
            total_precision += precision
    AP = total_precision / len(labels)
    return AP

# UNCOMMENT TO CALCULATE THE MODEL'S MAP score (REPLACE THE "modelFC" with appropriate variable name of the model)
# res = calculate_MAP(modelFC, test_hyponyms, test_hypernyms)


