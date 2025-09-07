from django.shortcuts import render, redirect
from .models import Review
from .forms import ReviewForm, UserRegistrationForm
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.conf import settings
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
# Create your views here.

def index(request):
    return render(request,'index.html')
def review_list(request):
  reviews = Review.objects.all().order_by('-created_at')
  return render(request,'review_list.html',{'reviews' : reviews})

@login_required
def review_create(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST, request.FILES)
        if form.is_valid():
            review = form.save(commit=False)
            review.user = request.user
            review.save()
            return redirect('review_list')
    else:
        form = ReviewForm()
    return render(request, 'review_form.html', {'form':form})

@login_required
def review_edit(request,review_id):
    review = get_object_or_404(Review, pk=review_id, user = request.user)
    if request.method=='POST':
        form = ReviewForm(request.POST, request.FILES, instance = review)
        if form.is_valid():
            review = form.save(commit = False)
            review.user = request.user
            review.save()
            return redirect('review_list')
    else:
        form = ReviewForm(instance=review)
    return render(request, 'review_form.html', {'form':form})
def review_delete(request, review_id):
    review = get_object_or_404(Review, pk=review_id, user=request.user)
    if request.method=='POST':
        review.delete()
        return redirect('review_list')
    return render(request, 'review_confirm_delete.html', {'review':review})

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    model_path = os.path.join(settings.BASE_DIR, 'review/models/model.pkl')
    tokenizer_path = os.path.join(settings.BASE_DIR, 'review/models/tokenizer.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(tokenizer_path, 'rb') as token_file:
        tokenizer = pickle.load(token_file)

load_model()


@login_required
def review_analyse(request, review_id):
    review = get_object_or_404(Review, id=review_id)
    review_text = review.text

    try:
        sequence = tokenizer.texts_to_sequences([review_text])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        sentiment_prediction = model.predict(padded_sequence)
        sentiment = "positive" if sentiment_prediction[0][0] > 0.5 else "negative"

        # Determine the color based on sentiment
        sentiment_color = "lime" if sentiment == "positive" else "red"

        # Return the result in the template or as JSON
        context = {
            'review': review,
            'sentiment': sentiment,
            'sentiment_color': sentiment_color  # Pass the color to the template
        }
        return render(request, 'review_analysis_result.html', context)

    except FileNotFoundError:
        return JsonResponse({'error': 'Model or tokenizer file not found'}, status=500)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def register(request):
    if request.method=="POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password1'])
            user.save()
            login(request, user)
            return redirect('review_list')
    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form':form})

@login_required
def review_list(request):
    reviews = Review.objects.all()
    
    # Handle search query
    query = request.GET.get('q')
    if query:
        # Case-insensitive search
        reviews = reviews.filter(movie_name__icontains=query)
        return render(request, 'review_search.html', {'reviews': reviews, 'query': query})
    
    return render(request, 'review_list.html', {'reviews': reviews})