Sure! Here's an expanded version of the README file, including information about Phase 1: Preprocessing and Boolean Retrieval, as well as Phase 2: Ranked Retrieval:

# Simple Information Retrieval System

This repository contains a simple information retrieval system that allows users to search for information within a collection of documents. The system provides two main phases: Preprocessing and Boolean Retrieval, and Ranked Retrieval.

## Table of Contents

- [Getting Started](#getting-started)
- [implementation](#implementationl)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with the Simple Information Retrieval System, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/amirerfantim/Simple-Information-Retrieval-System.git
   cd ir-project
   ```

2. Install the required dependencies:

   ```shell
   pip install hazm
   ```

3. Customize the collection of documents. Replace the sample documents in the `documents` directory with your own documents or update the existing documents.

## implementation
### Phase 1: Preprocessing and Boolean Retrieval

In this phase, the system performs preprocessing tasks on the documents to prepare them for retrieval. The following steps are involved:

1. Tokenization: The documents are split into individual tokens or words.

2. Stemming: The words are reduced to their base or root form. This helps to group similar words together and improve retrieval accuracy.

3. Stopword Removal: Common words that do not carry much meaning, such as "the," "is," and "and," are removed to reduce noise in the retrieval process.

4. Positional Indexing: The system creates a positional index, which is a data structure that stores the positions of words within each document. This allows for efficient phrase search.

To perform boolean retrieval, users can use certain operators and techniques:

- Phrases: Phrases can be declared using double quotes (""). For example, searching for "machine learning" will retrieve documents that contain the exact phrase "machine learning."

- NOT Operator: The NOT operator can be declared using an exclamation mark (!). For example, searching for "data !science" will retrieve documents that contain the word "data" but not the word "science."

The system uses the most frequent terms and phrases in the query to find relevant documents.

### Phase 2: Ranked Retrieval

In this phase, the system ranks the retrieved documents based on their relevance to the query. The following techniques are employed:

1. TF-IDF Weighting Scheme: The system uses the Term Frequency-Inverse Document Frequency (TF-IDF) weighting scheme to assign weights to terms in the documents and the query. TF-IDF gives higher weight to terms that appear frequently in a document but rarely in the entire collection. This helps to identify important terms that distinguish relevant documents.

2. Cosine Similarity: The system calculates the cosine similarity between the weighted query vector and the weighted document vectors. Cosine similarity measures the cosine of the angle

 between two vectors and determines their similarity. Higher cosine similarity values indicate greater relevance.

3. Jaccard Similarity Coefficient: The system also employs the Jaccard similarity coefficient to compare the query with the documents. The Jaccard similarity coefficient measures the similarity between two sets and is useful for handling multi-word queries. It calculates the ratio of the intersection of the query terms and the document terms to the union of both sets.

To improve the efficiency of the ranked retrieval process, the system uses champion lists. Champion lists are subsets of the documents that contain the most relevant documents for each term or phrase. By utilizing champion lists, the system reduces the number of documents that need to be compared, improving overall efficiency.



## License

The Simple Information Retrieval System is open source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as permitted by the license.

---

Thank you for using the Simple Information Retrieval System! If you have any questions or need further assistance, please don't hesitate to contact the repository owner or create an issue on the GitHub repository.
