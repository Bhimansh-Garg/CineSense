from .models import Review
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['text', 'movie_name', 'photo']

    def clean_text(self):
        text = self.cleaned_data.get('text') or ''
        normalized = text.strip()
        if not normalized:
            raise forms.ValidationError(
                'Review text cannot be empty or only whitespace.'
            )
        return normalized


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField()
    class Meta:
        model = User
        fields = {'username', 'email', 'password1', 'password2'}
