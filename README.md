# NarrRAG

NarrRAG is a modular narrative labeling framework that generates semantically precise and human-interpretable topic labels while minimizing human effort. NarrRAG uses standard topic model outputs to generate, validate, and refine narratives that serve as topic labels. The orchestrated RAG framework uses multiple retrieval strategies and chain-of-thought elements to provide high-quality output.

This repository serves as a demo implementation of NarrRAG using LangChain, LangGraph, Ollama (llama3.2) and Pydantic. We use a subsample (1,000 posts) of this X dataset on the U.S. election in 2024 (https://github.com/sinking8/x-24-us-election, see https://arxiv.org/abs/2411.00376 for the corresponding publication) to showcase basic functionality.

Three input documents are needed for NarrRAG:

1. one csv file with the text of each post ('Document') and the assigned topic ('Topic'),
2. one json file with representative topic keywords, and
3. one json file with news data

To obtain news data for our demo, we used the GNews package for Python (https://github.com/ranahaani/GNews).

For replication, we provide four files:
1. testdata_cleaned.csv, which is the input file for the BERTopic (https://github.com/MaartenGr/BERTopic/tree/master, https://arxiv.org/abs/2203.05794) model we need to run before NarrRAG can be applied (we tuned hyperparameter with OPTUNA, https://github.com/optuna/optuna),
2. testdata_seedtopics.csv, which equals the output of BERTopic modeling and includes the two columns necessary for NarrRAG 'Document' and 'Topic,
3. testdata_topic_keywords.json, which includes representative keywords for each topic, and
4. testdata_news.json, which consists of news content used for validation and obtained via GNews package.

The figure below presents the LangGraph visualization of NarrRAG's narrative extraction and validation pipeline:
<p align="center">
  <img src="/graph.png" width="600" alt="RAG Graph Visualization">
</p>

