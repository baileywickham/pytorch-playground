from model2vec import StaticModel

# Load a pretrained Model2Vec model
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Compute text embeddings
embeddings = model.encode(["man", "woman"])
print(embeddings, embeddings[0]*embeddings[1])
