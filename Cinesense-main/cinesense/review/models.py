import hashlib

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

from .image_utils import optimize_review_photo


class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField(max_length=400)
    movie_name = models.CharField(max_length=100, default='Empty')
    photo = models.ImageField(upload_to='photos/', blank=True, null=True)
    created_at = models.DateField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Cached sentiment for this review's current text (not a global cache).
    sentiment_label = models.CharField(max_length=16, blank=True, null=True)
    sentiment_score = models.FloatField(blank=True, null=True)
    sentiment_analyzed_at = models.DateTimeField(blank=True, null=True)
    sentiment_text_hash = models.CharField(max_length=64, blank=True, null=True)

    def __str__(self):
        return f'{self.user.username} - {self.text[:10]}'

    def compute_text_hash(self):
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()

    def clear_sentiment_cache(self):
        self.sentiment_label = None
        self.sentiment_score = None
        self.sentiment_analyzed_at = None
        self.sentiment_text_hash = None

    def has_valid_sentiment_cache(self):
        """True only when cached fields match this review's current text."""
        if self.sentiment_label not in ('positive', 'negative'):
            return False
        if self.sentiment_score is None:
            return False
        if not self.sentiment_text_hash:
            return False
        return self.sentiment_text_hash == self.compute_text_hash()

    def store_sentiment_cache(self, label, score):
        self.sentiment_label = label
        self.sentiment_score = float(score)
        self.sentiment_analyzed_at = timezone.now()
        self.sentiment_text_hash = self.compute_text_hash()
        # Avoid bumping updated_at merely because analysis ran.
        self.save(
            update_fields=[
                'sentiment_label',
                'sentiment_score',
                'sentiment_analyzed_at',
                'sentiment_text_hash',
            ]
        )

    def _photo_was_replaced(self) -> bool:
        """True when this save carries a new/replaced photo file."""
        if not self.photo:
            return False
        if not self.pk:
            return True
        previous_name = (
            type(self)
            .objects.filter(pk=self.pk)
            .values_list('photo', flat=True)
            .first()
        )
        if previous_name is None:
            return True
        return previous_name != self.photo.name

    def save(self, *args, **kwargs):
        update_fields = kwargs.get('update_fields')
        # Sentiment-cache and other partial saves must not touch the image.
        should_consider_photo = update_fields is None or 'photo' in update_fields
        if (
            should_consider_photo
            and self._photo_was_replaced()
            and not getattr(self, '_photo_optimization_done', False)
        ):
            optimized = optimize_review_photo(self.photo)
            if optimized is not None:
                # Assign in-memory content; a single super().save() persists it.
                self.photo = optimized
            self._photo_optimization_done = True

        super().save(*args, **kwargs)
