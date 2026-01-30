from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.core.validators import MinValueValidator


# class CustomUser(models.Model):
#     username = models.CharField(max_length=150, unique=True)
#     email = models.EmailField(unique=True)
#     age = models.PositiveIntegerField()
#     image = models.ImageField(upload_to='profile_pics/')
#     password = models.CharField(max_length=255)
    
#     def clean(self):
#         # Custom validation for age
#         if self.age <= 0:
#             raise ValidationError(_("Age must be a positive number."))

#     def __str__(self):
#         return self.username


class CustomUserManager(BaseUserManager):
    def create_user(self, username, email, age, password=None, image=None):
        if not email:
            raise ValueError("Users must have an email address")
        if not username:
            raise ValueError("Users must have a username")

        email = self.normalize_email(email)
        user = self.model(username=username, email=email, age=age, image=image)
        user.set_password(password)  # Hash password
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, age, password=None, image=None):
        user = self.create_user(username, email, age, password, image)
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user

class CustomUser(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    age = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    image = models.ImageField(upload_to="profile_pics/", null=True, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    groups = models.ManyToManyField(
        "auth.Group",
        related_name="customuser_set",  # Avoids conflict with default User model
        blank=True,
    )
    user_permissions = models.ManyToManyField(
        "auth.Permission",
        related_name="customuser_set",  # Avoids conflict with default User model
        blank=True,
    )

    objects = CustomUserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username", "age"]

    def __str__(self):
        return self.username
    



class Doctor(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)  # Store hashed passwords
    is_approved = models.BooleanField(default=False)  # Approval required
    name = models.CharField(max_length=255, default="Unknown Doctor")  # Default value added
    image = models.ImageField(upload_to='doctor_images/', blank=True, null=True)  # New field for profile image
    description = models.TextField(blank=True, null=True)  # New field for doctor description

    def __str__(self):
        return self.name


from django.db import models
from django.contrib.auth.models import User


from django.conf import settings
class VoicePrediction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    voice_file = models.FileField(upload_to='voice_uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.uploaded_at}"
class Appointment(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    patient_name = models.CharField(max_length=100)
    patient_age = models.IntegerField()
    appointment_date = models.DateField()
    appointment_time = models.TimeField()
    patient_email = models.EmailField(default='null')
    is_approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Appointment with {self.doctor.name} on {self.appointment_date}"