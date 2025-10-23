# Semantic Search System for Product Catalog

## Overview

This project implements and evaluates semantic search for product queries using multiple embedding models (all-MiniLM-L6-v2, BGE-small-en-v1.5, and E5-small-v2) and indexing strategies (Brute Force, IVF-PQ, and HNSW). The system was built on ~900K products from the Amazon ESCI dataset and evaluated on 50 sample queries using metrics including MRR, Precision@K, Recall@K, and NDCG@K. Results showed that E5-small-v2 achieved the best quality (NDCG@10: 0.380 with brute force), while IVF-PQ indexing provided ~20x speedup over brute force with only ~7% quality loss. HNSW metrics dropped and while various configurations of the index were tried performance could not be improved. Additional tuning or switching to faiss or milvus could provide the expected performance.

## Embeddings Generation

All fields from the dataset were used to try to capture as much information about the products as possible but it increases the amount of time it takes to generate the embeddings. The descriptions and the bullet points were truncated for long tail products in order to better fit into the sequence length. It would be interesting with additional time to look at the performance of indexes generated from less text and see how they compare to the embeddings generated in the notebook which includes all fields. 


## Data Quality Finding

During investigation of search results, it was discovered that there are products returned by the semantic search system that appear highly relevant but are not labeled in the ground truth dataset. This doesn't mean they are irrelevant—it simply means they weren't included in the manually labeled evaluation sample. For example, when searching for "alcohol prep pads", several top-ranked products with titles explicitly containing "Alcohol Prep Pads" were not in the ground truth set because they weren't sampled for labeling in the ESCI dataset. This indicates that the evaluation metrics (Precision, Recall, etc.) may underestimate the actual quality of the search system, as unlabeled-but-relevant products are incorrectly counted as false positives. It would be recommended to go through and clean up the ground truth before trying to further tune the search system.


## Setup

```bash
# Clone the Amazon ESCI data repo
# Available at: https://github.com/amazon-science/esci-data

# Install dependencies (requires Python 3.12)
# Or run first cell in the notebook
pip install -r requirements.txt

# If memory is a concern it might be helpful to save the subset of data for this project and then load only that into the kernel
# df_examples_products_filter = df_examples_products[(df_examples_products['esci_label']=='E') & (df_examples_products['product_locale']=='us')]
# df_examples_products_filter.to_parquet('./example_products.parquet')

# Run the main notebook to generate embeddings and indexes (or view saved output in cells)
jupyter notebook grainger_takehome.ipynb
```

**Note:** The LanceDB indexes and saved embeddings are not included in this repository as they are large artifacts. Running the notebook will regenerate them but the embeddings will take a good amount of time. Alternative you can view the results in the saved output of the cells.

