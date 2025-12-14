# Neural Networks with TensorFlow using PySpark

## 📌 Project Title

**Binary Classification using Feedforward Neural Network (PySpark + TensorFlow)**

---

## 📖 Project Description

This project demonstrates the implementation of a **binary classification model** using a **feedforward neural network with two hidden layers**. The solution integrates **PySpark** for scalable data generation and preprocessing, and **TensorFlow (Keras API)** for building, training, and evaluating the neural network.

The dataset is **synthetically generated within the code**, making the project fully self-contained without any external data dependency.

---

## 🎯 Objectives

* To generate a realistic binary classification dataset using PySpark
* To preprocess features using Spark ML utilities
* To design and train a feedforward neural network with two hidden layers
* To perform binary classification using TensorFlow optimizers
* To evaluate the performance of the neural network

---

## 🛠️ Technology Stack

* **Python 3.x**
* **PySpark** – Data generation and preprocessing
* **TensorFlow (Keras)** – Neural network modeling
* **NumPy** – Data conversion and handling

---

## 🧠 Neural Network Architecture

| Layer          | Neurons      | Activation Function |
| -------------- | ------------ | ------------------- |
| Input Layer    | 4 (features) | —                   |
| Hidden Layer 1 | 64           | ReLU                |
| Hidden Layer 2 | 32           | ReLU                |
| Output Layer   | 1            | Sigmoid             |

---

## 📂 Project Structure

```
├── main.py        # Complete PySpark + TensorFlow implementation
├── README.md     # Project documentation
```

---

## ⚙️ Dataset Details

* **Number of samples:** 1000
* **Number of features:** 4 (x1, x2, x3, x4)
* **Label:** Binary (0 or 1)
* **Label Generation Rule:**
  A weighted sum of features is compared against a threshold to assign class labels, simulating a real-world classification scenario.

---

## 🚀 How to Run the Project

1. Ensure PySpark and TensorFlow are installed
2. Start a Spark-supported Python environment
3. Run the main Python file:

   ```bash
   python main.py
   ```

---

## 📊 Model Training Details

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Epochs:** 20
* **Batch Size:** 32
* **Validation Split:** 20%

---

## ✅ Output

* Displays training and validation accuracy per epoch
* Prints final classification accuracy after evaluation

---

## 📝 Conclusion

This project successfully demonstrates how PySpark and TensorFlow can be combined to build a scalable and efficient neural network-based binary classification system. The synthetic dataset ensures reproducibility and eliminates dependency on external data sources.

---

## 👨‍🎓 Academic Use

✔ Suitable for lab records
✔ Mini-project submission ready
✔ Viva-friendly implementation

---

## 📬 Author

**Harshwardhan Singh Shekhawat**

---

> *"Learning neural networks becomes easier when scalability and deep learning frameworks work together."*
