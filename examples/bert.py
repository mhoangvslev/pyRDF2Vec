import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from transformers import TrainingArguments

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import BERT
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [entity for entity in train_data["bond"]]
train_labels = list(train_data["label_mutagenic"])

test_entities = [entity for entity in test_data["bond"]]
test_labels = list(test_data["label_mutagenic"])

entities = train_entities + test_entities
labels = train_labels + test_labels

embeddings, _ = RDF2VecTransformer(
    # Ensure random determinism for BERT.
    # Must be used with PYTHONHASHSEED.
    BERT(
        TrainingArguments(
            output_dir="./bert",
            overwrite_output_dir=True,
            num_train_epochs=3,
            warmup_steps=500,
            weight_decay=0.2,
            logging_dir="./logs",
            seed=RANDOM_STATE,
            dataloader_num_workers=2,
            prediction_loss_only=True,
        ),
    ),
    # Extract all walks with a maximum depth of 2 for each entity using two
    # processes and use a random state to ensure that the same walks are
    # generated for the entities.
    walkers=[RandomWalker(2, None, n_jobs=1, random_state=RANDOM_STATE)],
    verbose=1,
).fit_transform(
    KG(
        "samples/mutag/mutag.owl",
        skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
        literals=[
            [
                "http://dl-learner.org/carcinogenesis#hasBond",
                "http://dl-learner.org/carcinogenesis#inBond",
            ],
            [
                "http://dl-learner.org/carcinogenesis#hasAtom",
                "http://dl-learner.org/carcinogenesis#charge",
            ],
        ],
    ),
    entities,
)

train_embeddings = embeddings[: len(train_entities)]
test_embeddings = embeddings[len(train_entities) :]

# Fit a Support Vector Machine on train embeddings and pick the best
# C-parameters (regularization strength).
clf = GridSearchCV(
    SVC(random_state=RANDOM_STATE), {"C": [10 ** i for i in range(-3, 4)]}
)
clf.fit(train_embeddings, train_labels)

# Evaluate the Support Vector Machine on test embeddings.
predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
)
print(f"Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

# Reduce the dimensions of entity embeddings to represent them in a 2D plane.
X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
    train_embeddings + test_embeddings
)

# Define the color map.
colors = ["r", "g"]
color_map = {}
for i, label in enumerate(set(labels)):
    color_map[label] = colors[i]

# Set the graph with a certain size.
plt.figure(figsize=(10, 4))

# Plot the train embeddings.
plt.scatter(
    X_tsne[: len(train_entities), 0],
    X_tsne[: len(train_entities), 1],
    edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
    facecolors=[color_map[i] for i in labels[: len(train_entities)]],
)

# Plot the test embeddings.
plt.scatter(
    X_tsne[len(train_entities) :, 0],
    X_tsne[len(train_entities) :, 1],
    edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
    facecolors="none",
)

# Annotate few points.
plt.annotate(
    entities[25].split("/")[-1],
    xy=(X_tsne[25, 0], X_tsne[25, 1]),
    xycoords="data",
    xytext=(0.01, 0.0),
    fontsize=8,
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", facecolor="black"),
)
plt.annotate(
    entities[35].split("/")[-1],
    xy=(X_tsne[35, 0], X_tsne[35, 1]),
    xycoords="data",
    xytext=(0.4, 0.0),
    fontsize=8,
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", facecolor="black"),
)

# Create a legend.
plt.scatter([], [], edgecolors="r", facecolors="r", label="train -")
plt.scatter([], [], edgecolors="g", facecolors="g", label="train +")
plt.scatter([], [], edgecolors="r", facecolors="none", label="test -")
plt.scatter([], [], edgecolors="g", facecolors="none", label="test +")
plt.legend(loc="upper right", ncol=2)

# Display the graph with a title, removing the axes for
# better readability.
plt.title("pyRDF2Vec", fontsize=32)
plt.axis("off")
plt.show()
