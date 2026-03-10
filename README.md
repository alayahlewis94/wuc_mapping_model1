# wuc_mapping_model1
machine learning model that predicts wuc from text 

This project uses Natural Language Processing (NLP) and machine learning to predict aviation Work Unit Codes (WUC) from maintenance discrepancy narratives.

The goal is to improve maintenance data quality by identifying incorrectly coded Maintenance Action Forms (MAFs).

## Technologies Used

- Python
- Pandas
- NumPy
- SentenceTransformers
- Scikit-learn
- Cosine Similarity
- NLP

## How It Works

1. Maintenance discrepancy text is converted into semantic embeddings using SentenceTransformers.
2. Cosine similarity is used to compare discrepancy narratives with known WUC classifications.
3. The model recommends the most likely Work Unit Code based on semantic similarity.

## Potential Applications

- Aviation maintenance analytics
- Predictive maintenance
- Maintenance data quality improvement
- Fleet readiness analysis
