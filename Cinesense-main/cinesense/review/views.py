from django.shortcuts import render, redirect
from .models import Review
from .forms import ReviewForm, UserRegistrationForm
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.views import LoginView
from django.core.paginator import Paginator
from django.views.decorators.http import require_GET
from django.utils.decorators import method_decorator
from django.conf import settings
from django_ratelimit.decorators import ratelimit
import logging

from .ml_artifacts import ArtifactIntegrityError, load_paired_artifacts


logger = logging.getLogger(__name__)

REVIEW_LIST_PAGE_SIZE = 12
# movie_name max_length is 100; longer needles cannot match and only hurt SQLite scans.
REVIEW_SEARCH_QUERY_MAX_LEN = 100


def _review_analyse_rate(group, request):
    return getattr(settings, 'REVIEW_ANALYSE_RATELIMIT', '30/h')


def _login_rate(group, request):
    return getattr(settings, 'LOGIN_RATELIMIT', '5/m')


def normalize_review_search_query(raw_query):
    """Strip and bound the ?q= value used for movie_name__icontains search."""
    if raw_query is None:
        return ''
    return str(raw_query).strip()[:REVIEW_SEARCH_QUERY_MAX_LEN]


@method_decorator(
    ratelimit(key='ip', rate=_login_rate, method='POST', block=True),
    name='post',
)
@method_decorator(
    ratelimit(key='post:username', rate=_login_rate, method='POST', block=True),
    name='post',
)
class RateLimitedLoginView(LoginView):
    """Session login via Django's LoginView with POST throttling (IP + username)."""

    template_name = 'registration/login.html'


def index(request):
    return render(request, 'index.html')


@login_required
def review_list(request):
    # Default ordering is by primary key (no Meta.ordering on Review).
    reviews = Review.objects.all().order_by('pk')
    query = normalize_review_search_query(request.GET.get('q'))
    template_name = 'review_list.html'
    if query:
        # Case-insensitive substring match. On SQLite this is typically a table
        # scan (LIKE '%…%' cannot use a normal B-tree index). Acceptable for the
        # current SQLite deployment; see README "Database and search scalability"
        # for deferred PostgreSQL/pg_trgm work when the Review table grows.
        reviews = reviews.filter(movie_name__icontains=query)
        template_name = 'review_search.html'

    paginator = Paginator(reviews, REVIEW_LIST_PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(
        request,
        template_name,
        {
            'reviews': page_obj,
            'page_obj': page_obj,
            'query': query,
        },
    )


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
    return render(request, 'review_form.html', {'form': form})


@login_required
def review_edit(request, review_id):
    review = get_object_or_404(Review, pk=review_id, user=request.user)
    previous_text = review.text
    if request.method == 'POST':
        form = ReviewForm(request.POST, request.FILES, instance=review)
        if form.is_valid():
            review = form.save(commit=False)
            review.user = request.user
            if review.text != previous_text:
                # Text changed: never keep the previous sentiment as current.
                review.clear_sentiment_cache()
            review.save()
            return redirect('review_list')
    else:
        form = ReviewForm(instance=review)
    return render(request, 'review_form.html', {'form': form})


@login_required
def review_delete(request, review_id):
    review = get_object_or_404(Review, pk=review_id, user=request.user)
    if request.method == 'POST':
        review.delete()
        return redirect('review_list')
    return render(request, 'review_confirm_delete.html', {'review': review})


model = None
tokenizer = None
pad_sequences = None
artifact_manifest = None


def load_model():
    """Load verified Keras model + tokenizer pair once (or reuse test doubles)."""
    global model, tokenizer, pad_sequences, artifact_manifest
    if model is not None and tokenizer is not None and pad_sequences is not None:
        return
    model, tokenizer, pad_sequences, artifact_manifest = load_paired_artifacts()
    logger.info(
        'Sentiment artifacts ready version_id=%s',
        (artifact_manifest or {}).get('version_id'),
    )


def get_review_visible_to_user_or_404(user, review_id):
    """Return a review the user is allowed to view, or 404.

    Access model (from list/search UI and Review model):
    - Reviews are community-visible; there is no private/hidden flag.
    - Any authenticated user may view and analyze any existing review.
    - Ownership is enforced only for edit/delete, not for analysis.
    """
    if not user.is_authenticated:
        raise Http404('Authentication required to view reviews.')
    return get_object_or_404(Review, pk=review_id)


def extract_positive_probability(sentiment_prediction):
    """Return P(positive) from model.predict output, or None if unusable.

    The classifier ends in a sigmoid unit trained on positive=1 / negative=0,
    so values near 0 mean negative and near 1 mean positive. This is a class
    probability estimate, not a calibrated confidence interval.
    """
    try:
        score = float(sentiment_prediction[0][0])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if score != score or score in (float('inf'), float('-inf')):  # NaN / Inf
        return None
    return max(0.0, min(1.0, score))


def _sentiment_result_context(review, sentiment, prediction_score):
    sentiment_color = "lime" if sentiment == "positive" else "red"
    prediction_percent = int(round(float(prediction_score) * 100))
    return {
        'review': review,
        'sentiment': sentiment,
        'sentiment_color': sentiment_color,
        'prediction_score': float(prediction_score),
        'prediction_percent': prediction_percent,
    }


def _render_analyse_error(request, review, error_message, status=500):
    """HTML error page for the browser-facing analyse endpoint (no JSON)."""
    return render(
        request,
        'review_analysis_error.html',
        {
            'review': review,
            'error_message': error_message,
        },
        status=status,
    )


@login_required
@ratelimit(key='user', rate=_review_analyse_rate, method='GET', block=True)
@require_GET
def review_analyse(request, review_id):
    # Analyze any review the user may view (all reviews, once authenticated).
    # Rate limit is per authenticated user (key='user'); @login_required runs first.
    # Browser navigation endpoint: always return HTML (success or error).
    review = get_review_visible_to_user_or_404(request.user, review_id)

    try:
        if review.has_valid_sentiment_cache():
            context = _sentiment_result_context(
                review,
                review.sentiment_label,
                review.sentiment_score,
            )
            return render(request, 'review_analysis_result.html', context)

        load_model()
        sequence = tokenizer.texts_to_sequences([review.text])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        sentiment_prediction = model.predict(padded_sequence)
        prediction_score = extract_positive_probability(sentiment_prediction)
        if prediction_score is None:
            logger.error(
                'Invalid sentiment prediction for review_id=%s',
                review_id,
            )
            return _render_analyse_error(
                request,
                review,
                'Sentiment analysis could not interpret the model output. '
                'Please try again later.',
                status=500,
            )

        sentiment = "positive" if prediction_score > 0.5 else "negative"
        review.store_sentiment_cache(sentiment, prediction_score)
        context = _sentiment_result_context(review, sentiment, prediction_score)
        return render(request, 'review_analysis_result.html', context)

    except FileNotFoundError as exc:
        logger.error(
            'Missing sentiment artifacts for review_id=%s: %s',
            review_id,
            exc,
        )
        return _render_analyse_error(
            request,
            review,
            'Sentiment analysis is temporarily unavailable because required '
            'model files are missing.',
            status=500,
        )

    except ArtifactIntegrityError as exc:
        logger.error('Sentiment artifact integrity error: %s', exc)
        return _render_analyse_error(
            request,
            review,
            'Sentiment analysis is temporarily unavailable because model '
            'artifacts are missing or mismatched.',
            status=500,
        )

    except Exception:
        logger.exception(
            'Unexpected sentiment analysis failure for review_id=%s',
            review_id,
        )
        return _render_analyse_error(
            request,
            review,
            'Sentiment analysis failed unexpectedly. Please try again later.',
            status=500,
        )


def register(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # UserCreationForm.save() hashes password1; do not call set_password again.
            user = form.save()
            login(request, user)
            return redirect('review_list')
    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form': form})
