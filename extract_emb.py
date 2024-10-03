import torch


model_state_dict = torch.load('./sasrec/checkpoint/best.pt')

for key in model_state_dict.keys():
    print(key)


layer_weights = model_state_dict['embedding.weight'] 
import pickle

with open('./sasrec/sasrec_item.pkl', 'wb') as f:
    pickle.dump({'item_embedding': layer_weights}, f)