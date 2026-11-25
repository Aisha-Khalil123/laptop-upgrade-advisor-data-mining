# Smart Laptop Upgrade Advisor: Text & Graph Mining

Mining 19,545 laptop-user stories to group similar users, explain what each group needs, and find the most similar past cases for a new user, as groundwork for a "Smart Laptop Upgrade Advisor" that recommends the best first upgrade. Built for the *Data Mining* course at Al Hussein Technical University (2026).

## Problem

Retailers and IT support teams often give laptop upgrade advice by hand, using checklists or general rules, so advice is slow and inconsistent. Users describe their problems in free text, for example "slow with too many tabs" or "storage always full". This project tests whether data mining can:

1. Group similar user stories automatically (clustering)
2. Explain each group in plain language (text mining + LLM)
3. Find the past cases most similar to a new user's story (graph mining)

## Data

- `laptop_scenarios_story_text.csv`: 19,545 written user stories, each labelled with the user profile (student, office, gamer, designer, programmer), budget class, price tier, best first upgrade (RAM, Storage, GPU, No Upgrade Needed) and expected gain class
- The dataset was provided for the course, so it is not included in this repo.

## Notebooks

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `1_clustering.ipynb` | NLTK preprocessing (tokenisation, lowercasing, stop-word and punctuation removal, Porter stemming), TF-IDF, then PCA (74 components, 65% of variance). Compares 9 clustering setups and builds cluster persona cards |
| 2 | `2_text_mining_llm_insights.ipynb` | Extracts each cluster's top TF-IDF keywords and sends them to an LLM to generate a cluster name, a likely upgrade need and an explanation. Combines these into an "upgrade evidence sheet" per cluster |
| 3 | `3_graph_mining_similarity.ipynb` | Takes an input story, scores all 19,544 other stories with TF-IDF cosine similarity, builds a weighted similarity graph in NetworkX, and ranks cases with PageRank, degree, betweenness and closeness centrality |

## Results

### 1. Clustering comparison (external labels: `gain_class`)

| Model | ARI | AMI | Silhouette | Davies-Bouldin |
|-------|-----|-----|------------|----------------|
| **K-Means++ (k=3)** | **0.175** | **0.203** | 0.048 | 4.05 |
| K-Means random init (k=3) | 0.159 | 0.184 | 0.037 | 4.37 |
| K-Means++ (k=4) | 0.143 | 0.192 | 0.050 | 3.85 |
| Agglomerative, Ward (k=3) | 0.040 | 0.043 | 0.023 | 6.30 |
| Agglomerative, Complete / Average / Single (k=3) | ≈ 0 | ≈ 0 | 0.18–0.20 | 0.69–1.72 |
| DBSCAN (3 parameter trials) | ≈ 0 | ≤ 0.02 | n/a | n/a |

K-Means++ with k=3 matched the real labels best. The scores are modest overall. Many stories share the same template phrases, so clusters overlap. Complete, average and single linkage show higher silhouette scores, but they put almost every story into one cluster, and DBSCAN labelled over 96% of points as noise.

### 2. Cluster personas (K-Means++, k=3)

| Cluster | Size | Persona | Main profiles | Most common first upgrade |
|---------|------|---------|---------------|---------------------------|
| 0 | 1,656 | Low-Budget Frustrated Users | office, designer, gamer (all low budget) | RAM |
| 1 | 12,440 | Performance-Driven Power Users | gamer, designer, programmer | RAM, then Storage and GPU |
| 2 | 5,449 | Satisfied Office and Student Users | office, student | No Upgrade Needed |

### 3. Similarity graph

For the sample input story (a low-budget student whose best first upgrade is Storage), 33 stories had a similarity of 0.55 or more. The top 15 formed a graph with 16 nodes and 54 edges, and the highest similarity was 0.72. Centrality measures identified the most representative case (highest PageRank), the most connected case, a bridge case and an unusual case.

## Limitations and next steps

- The LLM labels are generated **only from keywords**, so they can be generic or misleading. For example, Cluster 0's keywords (`care`, `strict`, `hurt`) describe emotion, not hardware. Labels should be checked against the cluster statistics, as done in the persona cards.
- Low ARI/AMI shows that story text alone doesn't cleanly separate the gain classes. Next steps include sentence embeddings instead of TF-IDF, and a supervised classifier to predict the best first upgrade.

## Tools

Python, Pandas, NLTK, scikit-learn, SciPy, NetworkX, Matplotlib, OpenAI Python SDK (via OpenRouter), Google Colab

## How to run

1. Place `laptop_scenarios_story_text.csv` next to the notebooks.
2. For notebook 2, add your own OpenRouter API key in Colab's **Secrets** panel as `OPENROUTER_API_KEY`. No key is stored in this repo.
3. Run the notebooks in order: 1 → 2 → 3.

```bash
pip install pandas nltk scikit-learn scipy networkx matplotlib openai
```
