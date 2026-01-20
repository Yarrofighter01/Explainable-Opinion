# 🎬 Movie Review Sentiment Analyzer

An AI-powered application that analyzes movie reviews (IMDb) and provides **explainable sentiment predictions**.

It uses a fine-tuned **DistilBERT** model (91.6% Accuracy on IMDb) and leverages **XAI (Explainable AI)** techniques like **SHAP** and **Integrated Gradients** to show exactly *why* a review was classified as Positive or Negative.

## 🌟 Features
*   **High Accuracy**: 91.6% Accuracy on IMDb Movie Reviews.
*   **Visual Explanations**: See which words (e.g., "superb", "boring") influenced the decision.
*   **Interactive UI**: Built with Streamlit for easy usage.
*   **Optimized for Mac**: Runs efficiently on Apple Silicon (MPS).

## 🚀 Quick Start

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    ```bash
    streamlit run src/app.py
    ```

## 🛠 Project Structure
*   `src/app.py`: The web application entry point.
*   `src/explainer.py`: Logic for SHAP and Captum (Integrated Gradients).
*   `src/model_trainer.py`: Code used to train the model.
*   `src/evaluate.py`: Script to evaluate model performance on test sets.
*   `src/train.py`: Main training script.
*   `src/results_imdb`: (Generated after training) Contains the fine-tuned model.

## 📊 Training
The model was trained on the full IMDb dataset (25k reviews) using incremental learning and shuffling to prevent catastrophic forgetting.

To re-train the model yourself:
```bash
python src/run_incremental_imdb.py
```
