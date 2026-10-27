# Multi-Class Sentiment Analysis: A Comparative Study of Four Deep Learning Models

## 📌 Project Overview

This project presents a comparative study of four deep learning models for multi-class sentiment analysis:

- Simple Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- DistilBERT (pre-trained Transformer)

The objective is to classify text comments into three sentiment categories: negative, neutral, and positive. The models are trained and evaluated using a common dataset and a consistent train-validation-test split.

The project compares predictive performance using accuracy, precision, recall, F1-score, and confusion matrices. Training and inference times are also examined.

## 🎯 Objectives

- Perform text data cleaning and preprocessing.
- Prepare data for recurrent neural networks and Transformer models.
- Implement and train four deep learning models.
- Evaluate models using multiple classification metrics.
- Compare predictive performance and computational requirements.
- Identify the most effective model among the four evaluated approaches.

## 🛠️ Technologies Used

- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine learning:** scikit-learn
- **Deep learning:** TensorFlow, PyTorch
- **Transformer framework:** Hugging Face Transformers

## 📊 Dataset Description

The dataset was obtained from Kaggle.

**Dataset source:** [Sentiment Analysis Dataset — Kaggle](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset)

**Dataset file:** `sentiment_data.csv`

The original dataset contains 241,145 records with the following columns:

- `Comment`: Text comment used as the model input.
- `Sentiment`: Numeric sentiment label.

### Sentiment classes

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

### Data cleaning

The notebook removes missing comments, duplicate comments, and records outside the expected sentiment classes.

| Dataset statistic | Records |
|---|---:|
| Original records | 241,145 |
| Missing comments | 217 |
| Records after cleaning | 214,157 |

### Train-validation-test split

The cleaned dataset is divided using stratified sampling:

| Split | Records | Percentage |
|---|---:|---:|
| Training | 171,325 | 80% |
| Validation | 21,416 | 10% |
| Testing | 21,416 | 10% |
| **Total** | **214,157** | **100%** |

The dataset is not included in this repository. Download it from Kaggle and place `sentiment_data.csv` in the location expected by the notebook. Check the dataset's licence before redistributing it.

## ⚙️ Methodology

### 1. Data preprocessing

Text comments are converted to lowercase, URLs and HTML line-break tags are removed, and extra whitespace is normalized. Missing values and duplicate comments are removed before splitting the dataset.

### 2. Text representation

For Simple RNN, LSTM, and GRU, the text is converted into integer sequences using a Keras tokenizer. The vocabulary is limited to 20,000 words, and sequences are padded or truncated to 128 tokens.

DistilBERT uses its own pretrained tokenizer with a maximum sequence length of 128.

### 3. Model architectures

**Simple RNN:** Embedding layer → SimpleRNN → Dropout → Dense → Softmax.

**LSTM:** Embedding layer → LSTM → Dropout → Dense → Softmax.

**GRU:** Embedding layer → GRU → Dropout → Dense → Softmax.

**DistilBERT:** Pre-trained `distilbert-base-uncased` model fine-tuned for three-class classification.

### 4. Training configuration

The recurrent models use a vocabulary size of up to 20,000, embedding dimension of 128, 64 recurrent units, and a maximum sequence length of 128. Early stopping and learning-rate reduction callbacks are used.

DistilBERT is fine-tuned using PyTorch and the Hugging Face Transformers library, with a batch size of 16, two epochs, and a learning rate of 2e-5.

### 5. Evaluation

The models are evaluated on the held-out test set using accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, and confusion matrices. Training and inference times are also recorded.

## 📈 Results

The following test accuracies were obtained in the executed notebook:

| Model | Test Accuracy | Weighted F1-score |
|---|---:|---:|
| Simple RNN | 81.19% | 81.02% |
| LSTM | 84.02% | 83.93% |
| GRU | 83.95% | 83.85% |
| DistilBERT | 87.21% | 87.22% |

DistilBERT achieved the highest test accuracy among the four evaluated models. LSTM performed slightly better than GRU, while Simple RNN obtained the lowest accuracy in this experiment.

These results apply to this dataset, preprocessing pipeline, model configuration, and test split; they do not guarantee the same performance on other datasets.

## 🔍 Key Findings

- DistilBERT achieved the highest test accuracy in this experiment.
- LSTM and GRU produced similar classification performance.
- Simple RNN provided a baseline for comparison with more advanced sequence models.
- Confusion matrices help identify differences in classification performance across negative, neutral, and positive comments.
- Model selection should consider predictive performance as well as training cost and inference time.

## ▶️ How to Run the Project

### Google Colab

1. Open the notebook from this repository.
2. Open it in Google Colab.
3. Download `sentiment_data.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset).
4. Upload the CSV file to `/content/` in Colab.
5. Select a GPU runtime for the DistilBERT experiment.
6. Run the notebook cells in order.

### Local environment

Clone the repository:

```bash
git clone https://github.com/sanjayshrinivas07-blip/sentiment-analysis-model-comparison.git
cd sentiment-analysis-model-comparison
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch transformers accelerate jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Upload or place the dataset at the path expected by the notebook before running the cells. A compatible GPU is recommended for faster DistilBERT training.

## 📁 Project Structure

```text
sentiment-analysis-model-comparison/
├── Sentiment_Analysis_4_Models_Case_Study.ipynb
├── README.md
├── report.pdf                 # Optional
├── presentation.pptx          # Optional
└── results/                   # Optional plots and result files
```

The report, presentation, and results folder are optional. Update the filenames in this structure to match the files you actually upload.

## 🚧 Limitations

- The experiment uses one dataset and one fixed train-validation-test split.
- Results depend on preprocessing, model configuration, and training settings.
- DistilBERT requires more computational resources than the recurrent models in this experiment.
- Performance on unseen domains, languages, or differently distributed text has not been established.

## 🚀 Future Enhancements

- Evaluate the models on additional sentiment datasets.
- Tune hyperparameters and compare training configurations.
- Investigate class-specific performance and class imbalance.
- Explore explainability methods for model predictions.
- Develop a simple web application or API for sentiment prediction.
- Evaluate generalization using cross-domain or external test data.

## 👨‍💻 Author

**Sanjay Shrinivas**

M.Sc. Data Science Student

GitHub: [sanjayshrinivas07-blip](https://github.com/sanjayshrinivas07-blip)

## 📜 Academic Disclaimer

This project was developed as an academic case study to compare deep learning approaches for multi-class sentiment classification. Reported metrics reflect the executed experiment and should not be interpreted as a guarantee of real-world performance.

## 📄 Licence

The project currently has no specified open-source licence. Check the dataset's licence and the terms of any third-party model or code before redistributing them. Add a repository licence only after confirming that you have the right to license the included material.
