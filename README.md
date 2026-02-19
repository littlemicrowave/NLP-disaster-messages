## Project Outline
**[Report](https://github.com/littlemicrowave/NLP-disaster-messages/blob/main/report/report.pdf)**  
Project presents a comprehensive natural language processing pipeline prototype for multi-label classification and analysis of disaster response messages. Using the publicly available Disaster Response Messages dataset. A range of traditional and deep learning models were evaluated to assess their effectiveness in handling short, noisy, and imbalanced textual data. Classical linear models such as Logistic Regression and LinearSVC trained on TF-IDF and BoW representations demonstrated good baseline performance, achieving macro-F1 scores around 0.40–0.45 in multi-label classification of 35 targets. Deep neural networks, including dense multilayer perceptrons and bidirectional LSTM architectures were also examined along with BERT transformer. Beyond classification, auxiliary analyses such as Named Entity Recognition (NER), Topic Modeling with Latent Dirichlet Allocation (LDA) and BERTopic, along with Sentiment Analysis using VADER, and TextBlob provided interpretable insights into location, emotional tone, and emerging disaster themes. The integrated approach demonstrates that traditional vector-based models remain competitive for low-resource crisis text.

## How to Run

1. Use Python 3.10 or 3.11 (GPU optional for TensorFlow/PyTorch).
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Launch notebooks: `jupyter notebook` (or `jupyter lab`/Google Colab) and open the `.ipynb` files listed below.
5. Most notebooks load cached artifacts (e.g., `*_train.pkl`, `*_test.pkl`); rerun preprocessing only if you want to rebuild inputs.

## Task-to-File Map

- Data loading and preprocessing (task: cleaning, splitting, lemmatization; generates the preprocessing examples table in the report): `Loading_preprocessing.ipynb`
- Feature space analysis (task: BoW/TF-IDF/Word2Vec construction, frequency plots): `Classical_representation.ipynb`
- Classical baselines (task: SVM/Logistic Regression/Random Forest; produces the classification tables used in `report/main.tex`): `Classical_ML.ipynb`
- Deep models on cleaned text (task: dense MLP and BiLSTM with pretrained embeddings; tables/curves for report): `Deep_ML.ipynb`
- Deep models on semi-raw text (task: dense/BiLSTM variants): `Deep_ML_semi-raw.ipynb`
- Transformer classifier (task: DistilBERT on cleaned vs. semi-raw text; tables/plots for report): `DistilBERT.ipynb`, `DistilBERT_semi-raw.ipynb`
- Sentiment analysis (task: VADER/TextBlob sentiment scoring): `Sentiment_Analysis.ipynb`
- Named Entity Recognition (task: spaCy pipeline and visualizations): `NER.ipynb`
- Topic modeling (task: LDA/BERTopic plus topic visualizations): `Topic_Modeling.ipynb`
- Report sources (task: LaTeX + figures): `report/main.tex`, final PDF at `report/report.pdf`

If a single notebook covers multiple subtasks, relevant cells are commented with the task name (e.g., preprocessing, classifier training) to make navigation easier.

## Regenerating Tables in the Report

It is possible to run a notebooks is sequence, starting from the preprocessing, but in general:

- Preprocessing examples table (Section “Task 3: Feature Space Analysis”): rerun the tokenization/cleaning preview cells in `Loading_preprocessing.ipynb` and export the sample rows shown there.
- Classification result tables (SVM, Logistic Regression, Random Forest): rerun evaluation cells in `Classical_ML.ipynb` and use the displayed `classification_report` outputs (exported to LaTeX/Markdown in the notebook).
- Dense MLP and BiLSTM tables/curves: rerun `Deep_ML.ipynb` (and `Deep_ML_semi-raw.ipynb` for the semi-raw variant) to regenerate metrics and figures.
- DistilBERT tables/curves: rerun `DistilBERT.ipynb` (and `DistilBERT_semi-raw.ipynb`) to refresh the metrics reported in `report/main.tex`.
- Topic modeling figures/tables (LDA/BERTopic topics and term weights): rerun `Topic_Modeling.ipynb`; the notebook exports the topic lists/plots referenced in the report.

## Dataset Artifacts

The repository already includes serialized feature matrices and labels (e.g., `tfidf_train.pkl`, `tfidf_test.pkl`, `sense_vectors_train.pkl`, `sequences_train.pkl`, `y_train.pkl`). The notebooks load these files directly; delete them only if you intend to regenerate everything from raw data via `Loading_preprocessing.ipynb`.

# Research plan

Research question and novelty?

Focus can be put on fine-tuning the LLM and achieving a low level of model hallucinations when generating new disaster-relevant messages, if proper conditioning is done. For proper conditioning of disaster-relevant messages, we need to disseminate tweets, for instance, NER, Sentiment, disaster phase, and keywords + category (available); therefore, sophisticated feature engineering is required.

Focus can be put on an effective categorization pipeline (try to improve current F1-scores), which also requires sophisticated feature engineering employing disaster phase detection, keyword extraction, message sentiment, and named entities. We may probably try to use clustering-based methods, for instance, try to combine LDA/BERTopic/HDBSCAN clusters and word-frequency per category (some messages belong to multiple). For event detection, we can use our BERT model, which has 91% F1-score.

## Current situation.

Regarding message dissemination, we have: 

  1. NER.
  2. Sentiment Analysis.
  3. Topic modeling (BERTopic, LDA).
  4. Categories.
  5. Preprocessing stuff.
     
## TODO:

Depending on the research question (it was decided to move on with both)
We may use or even try to improve the Tweet4Act rule + PoS based approach on disaster phase detection. For instance, we can use rule based + PoS approach to detect easy, high-precision phase tags, and later perform semi-supervised training of a deep learning model.
* Keywords per category (ALMOST) Currently chi2 + idf reweightning is used. Benjamini–Hochberg FDR correction? conditional filtering, LODP?
* Keyword extractor from individual messages: RAKE?
* Finalize Tweet4Act-based high-precision phase labeler (POS/tense + temporal lexicons). (WORK IN PROGRESS)
* We need a validation set for phase/state evaluation, some can be labeled manually, for some we can use LLM, CoT approach
* Weakly supervised transformer to try generalize phase detection beyond rules.
* Finalize conditioning schema/features for classifier input (category + phase + entities + topic/keywords + urgency).
* Generation experiment and measure hallucination/constraint adherence.

