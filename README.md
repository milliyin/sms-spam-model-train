# 📩 SMS Spam Classifier

A simple spam detection project using **PyTorch**, **Scikit-learn**, and the **Hugging Face `sms_spam` dataset**. The model is trained to classify SMS messages as **spam** or **not spam** using a logistic regression-based neural network (`CircleModelV0`).

---
## 📁 Project Structure

```
Sms-Spam/
├── main.ipynb          # Training notebook
├── inference.py        # Inference script
├── full_model.pth      # Trained PyTorch model (full)
├── vectorizer.pkl      # CountVectorizer object (for text preprocessing)
└── README.md
```

---

## 🧠 Model Architecture (`CircleModelV0`)

A simple feed-forward neural network:
- `Input Layer`: 8713 features (from CountVectorizer)
- `Hidden Layer`: 64 neurons + ReLU
- `Output Layer`: 1 neuron (sigmoid for binary classification)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Sms-Spam.git
cd Sms-Spam
```

### 2. Install dependencies
```bash
pip install torch scikit-learn datasets huggingface_hub
```

### 3. Run Inference
```bash
python inference.py
```

---

## 🧪 Example Output

```bash
Input: you have won a free prize
Output: Spam
```

---

## 🗃️ Dataset

The model uses the public dataset [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) from Hugging Face. It contains 5,574 SMS messages labeled as `spam` or `ham`.

---

## 🧠 My Learning Journey

This project is part of my learning process on how to train machine learning models using my own code. Here’s what I’ve learned and implemented:

- ✅ Installed and configured libraries like `torch`, `scikit-learn`, and `datasets`.
- ✅ Downloaded and loaded the dataset using the Hugging Face Datasets library.
- ✅ Preprocessed the raw SMS data using `CountVectorizer` to convert text into numerical feature vectors.
- ✅ Converted those vectors to PyTorch tensors.
- ✅ Split the data into training and testing sets using `train_test_split`.
- ✅ Assigned input features (X: SMS prompts) and labels (y: 0 for ham, 1 for spam).
- ✅ Created a custom model class `CircleModelV0` using PyTorch's `nn.Module`.
- ✅ Wrote the training loop including forward pass, loss computation, backpropagation, and optimization.
- ✅ Evaluated the model using accuracy and tested it on new inputs.
- ✅ Saved and reloaded the model and vectorizer for inference.

---

## 🔐 Notes on PyTorch 2.6+

If you're using PyTorch 2.6+, the `torch.load()` call needs to allow loading full models with custom classes:
```python
model = torch.load("full_model.pth", weights_only=False)
```

Make sure the class `CircleModelV0` is defined or imported in your script before calling `torch.load`.

---


## 📬 Contact

Feel free to open issues or suggest improvements!  
Made with ❤️ for NLP enthusiasts.
