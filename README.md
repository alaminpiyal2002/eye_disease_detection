 This project is a web application for detecting eye diseases using a deep learning model. It uses a Convolutional Neural Network (CNN) trained with PyTorch to classify retinal images into six categories: Macular Scar, Myopia, Cataract, Diabetic Retinopathy, Glaucoma, and Normal. The application is built with Django and allows users to upload images to get predictions.

 ## Features
 - Upload retinal images for disease prediction.
 - Displays predicted disease and confidence score.
 - Built with Django for the web interface and PyTorch for the model.
 - Uses the `StudentCNN` model, optimized via knowledge distillation.

 ## Project Structure
 ```
 eye_disease_project/
 ├── eye_disease_app/
 │   ├── migrations/
 │   ├── templates/eye_disease_app/
 │   │   ├── index.html
 │   ├── model.py
 │   ├── views.py
 │   ├── ...
 ├── eye_disease_project/
 │   ├── settings.py
 │   ├── urls.py
 │   ├── ...
 ├── media/
 ├── static/
 ├── manage.py
 ├── student_model.pth
 ├── requirements.txt
 ├── README.md
 ├── .gitignore
 ```

 ## Setup Instructions
 1. **Clone the repository**:
    ```bash
    git clone https://github.com/alaminpiyal2002/eye-disease-detection.git
    cd eye-disease-detection
    ```
 2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
 3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
 4. **Run migrations**:
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```
 5. **Start the server**:
    ```bash
    python manage.py runserver
    ```
 6. Open `http://127.0.0.1:8000` in a browser.

 ## Usage
 - Upload a retinal image via the web interface.
 - View the predicted disease and confidence score.

 ## Model Details
 - **Model**: StudentCNN (distilled from CustomCNN).
 - **Classes**: Macular Scar, Myopia, Cataract, Diabetic Retinopathy, Glaucoma, Normal.
 - **Framework**: PyTorch.
 - **Input**: 224x224 RGB images.
 - **Training**: Trained on an augmented dataset with knowledge distillation.

 ## Requirements
 See `requirements.txt` for dependencies, including:
 - Django
 - PyTorch
 - Torchvision
 - Pillow
 - NumPy

 ## License
 This project is licensed under the MIT License. See `LICENSE` for details.

 ## Contributing
 Contributions are welcome! Please open an issue or submit a pull request.

 ## Contact
 For questions, contact [alamin876123@gmail.com].