# Character OCR Classifier (TensorFlow + CNN)

This project is a simple Optical Character Recognition (OCR) system built using TensorFlow and Keras. It trains a Convolutional Neural Network (CNN) to recognize **34 different characters** from grayscale images (28x28 pixels), using a dataset stored in the `character_ocr` folder.
Dataset Link: https://www.kaggle.com/datasets/inspiring-lab/nepali-number-plate-characters-dataset?resource=download
---


##  Model Architecture

- **Input:** 28x28 grayscale image
- **Layers:**
  - Flatten
  - Dense (256 units, ReLU)
  - Dense (128 units, ReLU)
  - Dropout (0.5)
  - Dense (Softmax for classification)

---

## Dependencies

Install the following Python packages before running:

```bash
pip install tensorflow numpy matplotlib seaborn pillow scikit-learn kagglehub
```

---

## Training the Model

Run the Jupyter Notebook or Python script containing the training code:

```python
model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)
```

Validation accuracy and training curves will be plotted after training.

---

## Evaluation

The notebook includes:
- Accuracy/loss plot
- Confusion matrix using `sklearn.metrics`
- Classification report

---


Make sure the image:
- Is grayscale or can be converted to grayscale
- Is resized to 28x28 pixels

---

## Output Example

Example accuracy plot and confusion matrix will be generated and shown automatically after training.

---

## 📬 Contact
 [Niranjan Luitel](https://github.com/LuitelN). Contributions and feedback are welcome!
