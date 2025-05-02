import torch
import torch.nn.functional as F
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image
import torchvision.transforms as transforms
import os
from .model import StudentCNN

# Define class names
CLASS_NAMES = ["Macular Scar", "Myopia", "Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'student_model.pth')
device = torch.device("cpu")  # Use CPU for free hosting
model = StudentCNN(num_classes=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Handle image upload
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_path = fs.path(filename)

            # Process the image
            image = Image.open(uploaded_file_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            image = image.to(device)

            # Make prediction
            with torch.no_grad():
                output = model(image)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item() * 100
                predicted_label = CLASS_NAMES[predicted_class]
                # Get all probabilities
                prob_dict = {CLASS_NAMES[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(probabilities[0])}

            # Prepare context for template
            context = {
                'uploaded_file_url': fs.url(filename),
                'prediction': predicted_label,
                'confidence': f"{confidence:.2f}%",
                'probabilities': prob_dict
            }
        except Exception as e:
            context = {'error': 'Invalid image file. Please upload a valid retinal image.'}
    return render(request, 'eye_disease_app/index.html', context)