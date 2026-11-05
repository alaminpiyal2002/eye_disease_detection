# Eye Disease Detection Using Deep Learning

A PyTorch-based deep learning system for **multi-class eye disease classification from retinal images**, integrated into a Django web application for image-based inference.

The project evaluates multiple convolutional and transformer-based architectures, investigates model compression through **pruning and knowledge distillation**, and incorporates **Grad-CAM and LIME** for model interpretability. A compact distilled CNN is ultimately integrated into the web application for efficient inference.

---

## Project Overview

This project focuses on automated classification of retinal images into six categories:

| Class |
|---|
| Macular Scar |
| Myopia |
| Cataract |
| Diabetic Retinopathy |
| Glaucoma |
| Normal |

The work covers the complete machine learning workflow, including dataset preparation, model development, comparative experimentation, model optimization, interpretability, and web-based deployment.

---

## Highlights

- Multi-class retinal image classification
- Custom CNN architecture developed from scratch
- Comparative evaluation of CNN and Vision Transformer architectures
- Knowledge distillation for lightweight inference
- Unstructured magnitude-based model pruning
- Grad-CAM and LIME-based interpretability
- Django-based image prediction interface
- PyTorch inference pipeline
- Reproducible training notebooks
- Project report containing experimental methodology and results

---

## Experimental Results

Multiple architectures were trained and evaluated on the same six-class classification task.

### Model Comparison

| Model | Accuracy |
|---|---:|
| ConvNeXt | 98.27% |
| EfficientNetV2 | 98.00% |
| MobileNetV3 | 97.80% |
| MNASNet | 97.77% |
| ResNeXt-50 | 97.50% |
| InceptionV3 | 97.47% |
| EfficientNet | 97.43% |
| MobileNetV2 | 96.97% |
| VGG16 | 96.47% |
| RegNet | 96.30% |
| ResNet50 | 95.67% |
| **Custom CNN** | **95.07%** |
| GoogleNet | 94.80% |
| DenseNet121 | 94.73% |
| Swin Transformer V2 | 97.10% |
| Vision Transformer | 96.10% |

The highest reported accuracy among the evaluated architectures was **98.27% using ConvNeXt**.

The custom CNN achieved **95.07% accuracy with approximately 2.10 million parameters** and was subsequently used as the teacher model for knowledge distillation.

---

## Model Compression

### Knowledge Distillation

A smaller `StudentCNN` was trained using knowledge transferred from the custom CNN teacher model.

| Model | Parameters | Size | Accuracy |
|---|---:|---:|---:|
| Custom CNN — Teacher | 2.10M | 7.42 MB | 95.07% |
| StudentCNN — Student | 0.85M | 3.02 MB | 93.50% |

Knowledge distillation reduced the model size by approximately **60%**, with a reported accuracy reduction of **1.5 percentage points**.

The resulting `StudentCNN` is the model integrated into the Django application.

### Pruning

Unstructured magnitude-based pruning was evaluated at:

- 20% sparsity
- 50% sparsity
- 70% sparsity

At 70% sparsity, the model size was reduced from approximately **7.42 MB to 2.84 MB** after pruning and fine-tuning, with a reported accuracy of approximately **94.50%**.

---

## Interpretability

Two post-hoc explainability techniques were explored to analyze model predictions.

### Grad-CAM

Grad-CAM produces class-specific activation heatmaps highlighting image regions contributing to a prediction.

### LIME

LIME provides local explanations by identifying image regions that influence an individual prediction.

These methods were used to provide visual insight into model behavior beyond the predicted class and confidence score.

---

## Dataset

The dataset contains six eye-disease categories.

### Original Distribution

| Class | Images |
|---|---:|
| Macular Scar | 1,000 |
| Myopia | 1,012 |
| Cataract | 1,038 |
| Diabetic Retinopathy | 1,098 |
| Glaucoma | 1,007 |
| Normal | 1,074 |

The images were augmented to **2,500 samples per class**.

### Final Dataset

| Split | Images |
|---|---:|
| Training | 10,500 |
| Validation | 1,500 |
| Testing | 3,000 |
| **Total** | **15,000** |

All images were resized to **224 × 224 pixels** for model development.

> The dataset is not stored in this repository because of its size.  
> **Dataset:** [Google Drive — to be added]

---

## Repository Structure

```text
eye_disease_detection/
│
├── eye_disease_app/
│   ├── migrations/
│   ├── templates/
│   │   └── eye_disease_app/
│   ├── model.py
│   ├── views.py
│   └── ...
│
├── eye_disease_project/
│   ├── settings.py
│   ├── urls.py
│   └── ...
│
├── notebooks/
│   ├── Custom_CNN.ipynb
│   ├── ConvNext.ipynb
│   ├── DenseNet121.ipynb
│   ├── EfficientNet.ipynb
│   ├── EfficientNetV2.ipynb
│   ├── GoogleNet.ipynb
│   ├── InceptionV3.ipynb
│   ├── MnasNet.ipynb
│   ├── MobileNetV2.ipynb
│   ├── MobileNetV3.ipynb
│   ├── RegNet.ipynb
│   ├── ResNet50.ipynb
│   ├── ResNeXt-50.ipynb
│   ├── SwingTransformerV2.ipynb
│   ├── VGG16.ipynb
│   └── VisionTransformer_base.ipynb
│
├── report/
│   └── Eye_Disease_Detection_Project_Report.pdf
│
├── media/
├── manage.py
├── student_model.pth
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Torchvision |
| Image Processing | Pillow, NumPy |
| Web Framework | Django |
| Frontend | HTML, CSS, Django Templates |
| Model Interpretation | Grad-CAM, LIME |
| Version Control | Git, GitHub |

---

## Running the Application

### Prerequisites

- Python 3.9+
- Git
- pip
- Virtual environment

### 1. Clone the repository

```bash
git clone https://github.com/alaminpiyal2002/eye_disease_detection.git
cd eye_disease_detection
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Start the application

```bash
python manage.py runserver
```

Open the application at:

```text
http://127.0.0.1:8000/
```

---

## Inference Workflow

The deployed application follows a simple inference pipeline:

```text
Retinal Image
      │
      ▼
Image Upload
      │
      ▼
Preprocessing
      │
      ▼
StudentCNN
      │
      ▼
Class Prediction
      │
      ▼
Predicted Disease + Confidence
```

The application uses the trained `StudentCNN` model stored in `student_model.pth`.

---

## Training Configuration

The primary training experiments used:

| Parameter | Configuration |
|---|---|
| Input Resolution | 224 × 224 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Maximum Epochs | 50 |
| Early Stopping Patience | 5 |
| LR Scheduler Factor | 0.1 |
| LR Scheduler Patience | 3 |

---

## Project Resources

### Training Notebooks

The [`notebooks/`](notebooks/) directory contains the experimental implementations used to evaluate the different architectures.

### Project Report

The complete project report is available at:

[`report/Eye_Disease_Detection_Project_Report.pdf`](report/Eye_Disease_Detection_Project_Report.pdf)

### Dataset

The dataset is hosted externally due to its size.

**Google Drive:** `To be added`

---

## Limitations and Disclaimer

This project is intended for **research, educational, and demonstration purposes**.

The model's predictions should not be interpreted as medical diagnoses or used as a substitute for professional ophthalmological assessment.

Model performance may vary with changes in image quality, acquisition conditions, patient populations, and datasets. The reported experimental results should therefore be interpreted within the scope of the dataset and evaluation procedure used in this project.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Al Amin**

For questions, collaboration, or feedback:

**Email:** [alamin876123@gmail.com](mailto:alamin876123@gmail.com)

---

## Acknowledgements

This project was developed as an academic deep learning project exploring retinal image classification, model optimization, and explainable AI.
