import os
import json
import random
import pickle
import numpy as np
import librosa
import gdown
import glob
from io import BytesIO
from django.shortcuts import render, redirect, HttpResponse
from django.http import JsonResponse
from django.conf import settings
from django.contrib import messages
from django.views.generic import DetailView
from reportlab.pdfgen import canvas
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.decorators import login_required
from .forms import (
    SignupForm, 
    LoginForm,
    DoctorRegistrationForm, 
    DoctorLoginForm,
    VoiceUploadForm
)
from .models import CustomUser, Doctor, VoicePrediction,Appointment

# ======================
# MODEL CONFIGURATION
# ======================
MODEL_PATH = os.path.join(settings.BASE_DIR, 'best_model.keras')
MODEL_URL = "https://drive.google.com/uc?id=1T2LVJDstrxgKeLBera79PBGX384vil7A"
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, "label_encoder.pkl")

def initialize_ml_components():
    """Initialize ML model and label encoder with proper error handling"""
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading model...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            print(f"Model download failed: {str(e)}")
            return None, None

    # Load model with LSTM compatibility fix
    try:
        model = load_model(
            MODEL_PATH,
            custom_objects={'LSTM': LSTM},
            compile=False
        )
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        return model, le
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return None, None

# Initialize ML components when module loads
best_model, label_encoder = initialize_ml_components()

# ======================
# HELPER FUNCTIONS
# ======================
def handle_uploaded_file(uploaded_file):
    """Save uploaded file to temp location and return path"""
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, 'wb+') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    
    return temp_path

def pad_features(features, max_length):
    """Pad features to the same length"""
    padded_features = []
    for f in features:
        if len(f) < max_length:
            pad_width = max_length - len(f)
            padded_f = np.pad(f, ((0, pad_width), (0, 0)), mode='constant')
        else:
            padded_f = f[:max_length, :]
        padded_features.append(padded_f)
    return np.array(padded_features)

def extract_features_for_prediction(file_path):
    """Extract features in the format expected by the model"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T
        
        return mfcc, chroma, mel, spectral_contrast
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        raise

def generate_prediction_report(name, age, prediction):
    """Generate PDF report with proper formatting"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    
    # Title
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, 800, "COPD PREDICTION REPORT")
    p.line(100, 795, 500, 795)
    
    # Patient Info
    p.setFont("Helvetica", 12)
    p.drawString(100, 750, f"Patient Name: {name}")
    p.drawString(100, 730, f"Age: {age}")
    
    # Prediction Result
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 700, "Prediction Result:")
    p.setFont("Helvetica", 12)
    p.drawString(120, 680, f"{prediction}")
    
    # Footer
    p.setFont("Helvetica-Oblique", 10)
    p.drawString(100, 650, "Note: This result should be interpreted by a medical professional.")
    
    p.showPage()
    p.save()
    
    # Get the PDF data and close the buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return BytesIO(pdf_data)

# ======================
# CORE VIEWS
# ======================
def home(request):
    return render(request, 'home.html')

def signup(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        age = request.POST.get("age")
        image = request.FILES.get("image")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect("signup")

        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect("signup")

        user = CustomUser.objects.create_user(
            username=username, email=email, age=age, password=password, image=image
        )
        messages.success(request, "Account created successfully! You can now log in.")
        return redirect("login")

    return render(request, "signup.html")

def signin(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            try:
                user = CustomUser.objects.get(username=username)
                if check_password(password, user.password):
                    request.session['user_id'] = user.id
                    request.session['user_type'] = 'patient'
                    messages.success(request, f"Welcome back, {username}!")
                    return redirect('landing')
                messages.error(request, "Invalid password")
            except CustomUser.DoesNotExist:
                messages.error(request, "User not found")
    else:
        form = LoginForm()
    return render(request, 'signin.html', {'form': form})

def landing(request):
    if 'user_id' not in request.session:
        return redirect('login')
    
    user_id = request.session['user_id']
    if request.session.get('user_type') == 'patient':
        user = CustomUser.objects.get(id=user_id)
    else:
        user = Doctor.objects.get(id=user_id)
    
    return render(request, 'landing.html', {'user': user})

# ======================
# DOCTOR VIEWS
# ======================
from django.contrib.auth.hashers import make_password

def register_doctor(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        name = request.POST.get('name')
        image = request.FILES.get('image')
        description = request.POST.get('description')

        # Check if username or email already exists
        if Doctor.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Choose a different one.")
            return redirect('register_doctor')

        if Doctor.objects.filter(email=email).exists():
            messages.error(request, "Email already registered. Try logging in.")
            return redirect('register_doctor')

        # Hash the password before saving
        hashed_password = make_password(password)

        # Create doctor (with approval pending)
        doctor = Doctor.objects.create(
            username=username,
            email=email,
            password=hashed_password,  # Store the hashed password
            name=name,
            image=image,
            description=description,
            is_approved=False  # Requires admin approval
        )

        messages.success(request, "Registration successful! Please wait for admin approval.")
        return redirect('doctor_login')

    return render(request, 'dr_register.html')



def doctor_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            doctor = Doctor.objects.get(username=username)
        except Doctor.DoesNotExist:
            messages.error(request, "Username does not exist.")
            return redirect('doctor_login')

        # Check if password is correct
        if check_password(password, doctor.password):
            if doctor.is_approved:  # Ensure the doctor is approved
                request.session['doctor_id'] = doctor.id  # Store doctor ID in session
                messages.success(request, "Login successful!")
                return redirect('doctor_dashboard')  # Redirect to doctor's dashboard
            else:
                messages.error(request, "Your account is pending approval.")
                return redirect('doctor_login')
        else:
            messages.error(request, "Invalid password.")
            return redirect('doctor_login')

    return render(request, 'dr_login.html')

def doctor_dashboard(request):
    doctor_id = request.session.get('doctor_id')
    if not doctor_id:  
        return redirect("doctor_login")

    try:
        doctor = Doctor.objects.get(id=doctor_id)
        approved_doctors = Doctor.objects.filter(is_approved=True)
        non_approved_doctors = Doctor.objects.filter(is_approved=False)
        return render(request, "doctor_dashboard.html", {
            "doctor": doctor,
            "approved_doctors": approved_doctors,
            "non_approved_doctors": non_approved_doctors
        })
    except Doctor.DoesNotExist:
        messages.error(request, "Doctor account not found.")
        return redirect("doctor_login")

def doctor_logout(request):
    request.session.flush()
    return redirect('home')

def doctor_list(request):
    doctors = Doctor.objects.filter(is_approved=True)
    return render(request, 'doctor.html', {'doctors': doctors})

class DoctorDetailView(DetailView):
    model = Doctor
    template_name = 'doctor_detail.html'
    context_object_name = 'doctor'

# ======================
# PREDICTION VIEWS
# ======================
def prediction_view(request):
    if request.method == 'POST':
        form = VoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save the uploaded file temporarily
                temp_path = handle_uploaded_file(form.cleaned_data['voice'])
                
                # Extract features in the correct format
                mfcc, chroma, mel, spectral_contrast = extract_features_for_prediction(temp_path)
                
                # Prepare features for prediction
                features = [
                    np.expand_dims(pad_features([mfcc], 2000), axis=-1),
                    np.expand_dims(pad_features([chroma], 2000), axis=-1),
                    np.expand_dims(pad_features([mel], 2000), axis=-1),
                    np.expand_dims(pad_features([spectral_contrast], 2000), axis=-1)
                ]
                
                # Make prediction
                print('hiii2')
                predictions = best_model.predict(features)
                predicted_label = np.argmax(predictions, axis=1)
                prediction_result = label_encoder.inverse_transform(predicted_label)[0]
                
                # Clean up the temp file
                os.remove(temp_path)
                
                # Generate report
                name = request.POST.get('name', 'User')
                age = request.POST.get('age', '')
                pdf_buffer = generate_prediction_report(name, age, prediction_result)
                
                # Create response with proper headers
                response = HttpResponse(content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="COPD_Report_{name}.pdf"'
                response.write(pdf_buffer.getvalue())
                pdf_buffer.close()
                return response
                
            except Exception as e:
                messages.error(request, f"Prediction failed: {str(e)}")
                return redirect('prediction')
        else:
            messages.error(request, "Invalid form submission")
            return redirect('prediction')
    else:
        form = VoiceUploadForm()
    return render(request, 'prediction.html', {'form': form})
    

# ======================
# CHATBOT & MISC VIEWS
# ======================
import json
import random
from django.http import JsonResponse
from django.shortcuts import render
from .models import Doctor

def chatbot_response(request):
    user_message = request.GET.get('message', '').lower()
    
    # Basic responses
    responses = {
        'hello': 'Hello! How can I assist you today?',
        'hi': 'Hi there! Need help finding a doctor?',
        'help': 'I can help you find a doctor. Just ask!',
        'bye': 'Goodbye! Take care.',
    }
    
    if user_message in responses:
        return JsonResponse({'response': responses[user_message]})
    
    if 'doctor' in user_message:
        available_doctors = Doctor.objects.filter(is_available=True)
        if available_doctors.exists():
            selected_doctor = random.choice(available_doctors)
            return JsonResponse({'response': f'The best available doctor is {selected_doctor.name} from {selected_doctor.department}.'})
        else:
            return JsonResponse({'response': 'Sorry, no doctors are available right now.'})
    
    return JsonResponse({'response': "I'm not sure how to respond to that."})

def chatbot(request):
    doctors = Doctor.objects.filter(is_approved=True)  
    doctors_json = json.dumps(list(doctors.values('id', 'name')))
    return render(request, 'chatbot.html', {'doctors_json': doctors_json})

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def get_doctors(request):
    doctors = Doctor.objects.filter(is_approved=True).values('id', 'name', 'department')
    return JsonResponse({'doctors': list(doctors)})
# =======================
#learn how it works
#=========================

from django.shortcuts import render

def how_it_works(request):
    return render(request, 'how_it_works.html')  # Render the new template

import os
import numpy as np
import pickle
import librosa
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from django.core.files.storage import default_storage

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model/best_model.keras")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "ml_model/label_encoder.pkl")

model = load_model(MODEL_PATH)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Function to Extract Features from Audio
def extract_features(file_path, max_length=2000):
    y, sr = librosa.load(file_path, sr=22050)  

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T

    def pad_or_truncate(feature, max_length):
        if feature.shape[0] < max_length:
            pad_width = max_length - feature.shape[0]
            return np.pad(feature, ((0, pad_width), (0, 0)), mode='constant')
        else:
            return feature[:max_length, :]

    mfcc = np.expand_dims(pad_or_truncate(mfcc, max_length), axis=-1)
    chroma = np.expand_dims(pad_or_truncate(chroma, max_length), axis=-1)
    mel = np.expand_dims(pad_or_truncate(mel, max_length), axis=-1)
    spectral_contrast = np.expand_dims(pad_or_truncate(spectral_contrast, max_length), axis=-1)

    return np.array([mfcc]), np.array([chroma]), np.array([mel]), np.array([spectral_contrast])

@csrf_exempt
def predict_disease(request):
    if request.method == "POST" and request.FILES.get("voice"):
        audio_file = request.FILES["voice"]

      
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT) 

      
        file_name = default_storage.save("temp_audio.wav", audio_file) 

        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        print("File saved at:", file_path) 
        
        print('hiii1')

        try:
            # Extract features
            mfcc, chroma, mel, spectral_contrast = extract_features(file_path)

            # Make predictions
            predictions = model.predict([mfcc, chroma, mel, spectral_contrast])
            predicted_class = np.argmax(predictions, axis=1)
            predicted_label = le.inverse_transform(predicted_class)[0]
            print('hiii1')
            print(predicted_label)


            # Clean up
            os.remove(file_path)

            return JsonResponse({"prediction": predicted_label})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
def book_appointment(request):
    doctors = Doctor.objects.all()  # Fetch only available doctors

    if request.method == "POST":
        patient_name = request.POST.get("name")
        patient_age = request.POST.get("age")
        doctor_id = request.POST.get("doctor")
        appointment_date = request.POST.get("date")
        appointment_time = request.POST.get("time")

        try:
            doctor = Doctor.objects.get(id=doctor_id)
            Appointment.objects.create(
                doctor=doctor,
                patient_name=patient_name,
                patient_age=patient_age,
                appointment_date=appointment_date,
                appointment_time=appointment_time
            )
            return JsonResponse({"message": "Appointment booked successfully!"})
        except Doctor.DoesNotExist:
            return JsonResponse({"error": "Doctor not found"}, status=400)

    return render(request, "appointment_booking.html", {"doctors": doctors})


def confirmation_page(request, appointment_id):
    appointment = Appointment.objects.get(id=appointment_id)
    return render(request, "confirmation.html", {"appointment": appointment})
from django.shortcuts import render, get_object_or_404


def doctor_detail(request, doctor_id):
    doctor = get_object_or_404(Doctor, id=doctor_id)
    appointments = Appointment.objects.filter(doctor=doctor).order_by("appointment_date", "appointment_time")

    return render(request, "doctor_detail.html", {"doctor": doctor, "appointments": appointments})

def appointment_schedule(request, doctor_id):
    doctor = get_object_or_404(Doctor, id=doctor_id)
    appointments = Appointment.objects.filter(doctor=doctor).order_by("appointment_date", "appointment_time")

    return render(request, "appointments_list.html", {"doctor": doctor, "appointments": appointments})

from django.core.mail import send_mail
from django.shortcuts import get_object_or_404

# ======================
# EMAIL ON APPROVAL
# ======================
def send_appointment_approval_email(appointment):
    subject = "Your Appointment Has Been Approved"
    message = (
        f"Dear {appointment.patient_name},\n\n"
        f"Your appointment with  {appointment.doctor.name} "
        f"on {appointment.appointment_date} at {appointment.appointment_time} "
        f"has been approved.\n\n"
        f"Thank you for using our COPD Prediction System."
    )
    recipient_email = appointment.patient_email
    send_mail(subject, message, settings.EMAIL_HOST_USER, [recipient_email], fail_silently=False)

# ======================
# DOCTOR APPROVES APPOINTMENT
# ======================
def approve_appointment(request, appointment_id):
    appointment = get_object_or_404(Appointment, id=appointment_id)
    appointment.is_approved = True
    appointment.save()

    send_appointment_approval_email(appointment)

    messages.success(request, "Appointment approved and email sent to patient.")
    return redirect("appointment_schedule", doctor_id=appointment.doctor.id)