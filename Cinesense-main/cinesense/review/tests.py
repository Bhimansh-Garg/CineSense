from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.urls import reverse

from .models import Review

import os


# Minimal valid 1x1 PNG so analysis templates can resolve review.photo.url
_TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
    b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
    b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
)


def mock_sentiment(predict_return=None, predict_side_effect=None):
    """Patch ML globals so unit tests never load TensorFlow artifacts."""
    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    model = MagicMock()
    if predict_side_effect is not None:
        model.predict.side_effect = predict_side_effect
    else:
        model.predict.return_value = predict_return if predict_return is not None else [[0.9]]
    pad = MagicMock(return_value=[[0] * 200])
    return patch.multiple(
        'review.views',
        tokenizer=tokenizer,
        model=model,
        pad_sequences=pad,
    )


class ReviewAnalyseAuthorizationTests(TestCase):
    """Analysis is allowed for any review visible to the authenticated user.

    Reviews are community-visible (no private flag). Edit/delete remain owner-only.
    """

    def setUp(self):
        self.client = Client()
        self.owner = User.objects.create_user(username='owner', password='pass12345')
        self.other = User.objects.create_user(username='other', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.owner,
            text='This movie was fantastic and thrilling.',
            movie_name='Test Movie',
            photo=photo,
        )
        self.url = reverse('review_analyse', args=[self.review.pk])
        self.missing_url = reverse('review_analyse', args=[999999])

    def test_owner_can_analyse_own_review(self):
        self.client.login(username='owner', password='pass12345')
        with mock_sentiment():
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'positive')

    def test_other_authenticated_user_can_analyse_visible_review(self):
        # Matches UI: Analyze is offered for every review to any logged-in user.
        self.client.login(username='other', password='pass12345')
        with mock_sentiment():
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.review.text)

    def test_unauthenticated_user_is_redirected_to_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_nonexistent_review_returns_404(self):
        self.client.login(username='owner', password='pass12345')
        with mock_sentiment():
            response = self.client.get(self.missing_url)
        self.assertEqual(response.status_code, 404)

    def test_post_method_not_allowed(self):
        self.client.login(username='owner', password='pass12345')
        with mock_sentiment():
            response = self.client.post(self.url)
        self.assertEqual(response.status_code, 405)

    def test_no_private_review_flag_on_model(self):
        field_names = {f.name for f in Review._meta.get_fields()}
        self.assertNotIn('is_private', field_names)
        self.assertNotIn('visibility', field_names)
        self.assertNotIn('is_public', field_names)


class ReviewAnalyseBehaviorTests(TestCase):
    """Happy path and error handling for sentiment analysis (mocked ML)."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='analyst', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.user,
            text='Analysis behavior review.',
            movie_name='Behavior Movie',
            photo=photo,
        )
        self.url = reverse('review_analyse', args=[self.review.pk])
        self.client.login(username='analyst', password='pass12345')

    def test_happy_path_positive_sentiment(self):
        with mock_sentiment(predict_return=[[0.92]]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'positive')
        self.assertContains(response, '92% positive probability')
        self.assertEqual(response.context['prediction_score'], 0.92)
        self.assertEqual(response.context['prediction_percent'], 92)
        self.assertTemplateUsed(response, 'review_analysis_result.html')

    def test_happy_path_negative_sentiment(self):
        with mock_sentiment(predict_return=[[0.1]]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'negative')
        self.assertContains(response, '10% positive probability')
        self.assertEqual(response.context['prediction_percent'], 10)

    def test_near_threshold_prediction_score_formatting(self):
        with mock_sentiment(predict_return=[[0.51]]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['sentiment'], 'positive')
        self.assertEqual(response.context['prediction_percent'], 51)
        self.assertContains(response, 'positive — 51% positive probability')

        # Clear cache so a new inference path is exercised for 0.49.
        self.review.refresh_from_db()
        self.review.clear_sentiment_cache()
        self.review.save(
            update_fields=[
                'sentiment_label',
                'sentiment_score',
                'sentiment_analyzed_at',
                'sentiment_text_hash',
            ]
        )

        with mock_sentiment(predict_return=[[0.49]]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['sentiment'], 'negative')
        self.assertEqual(response.context['prediction_percent'], 49)
        self.assertContains(response, 'negative — 49% positive probability')

    def test_invalid_prediction_returns_500_html(self):
        with mock_sentiment(predict_return=[[float('nan')]]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response['Content-Type'].split(';')[0], 'text/html')
        self.assertTemplateUsed(response, 'review_analysis_error.html')
        self.assertContains(
            response,
            'could not interpret the model output',
            status_code=500,
        )
        self.assertNotContains(response, 'nan', status_code=500)

        self.review.refresh_from_db()
        self.review.clear_sentiment_cache()
        self.review.save(
            update_fields=[
                'sentiment_label',
                'sentiment_score',
                'sentiment_analyzed_at',
                'sentiment_text_hash',
            ]
        )
        with mock_sentiment(predict_return=[]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response['Content-Type'].split(';')[0], 'text/html')

    def test_extract_positive_probability_unit(self):
        from review.views import extract_positive_probability

        self.assertEqual(extract_positive_probability([[0.83]]), 0.83)
        self.assertEqual(extract_positive_probability([[0.0]]), 0.0)
        self.assertEqual(extract_positive_probability([[1.0]]), 1.0)
        self.assertIsNone(extract_positive_probability(None))
        self.assertIsNone(extract_positive_probability([]))
        self.assertIsNone(extract_positive_probability([[float('nan')]]))

    def test_nonexistent_review_returns_404(self):
        missing = reverse('review_analyse', args=[999999])
        with mock_sentiment():
            response = self.client.get(missing)
        self.assertEqual(response.status_code, 404)

    def test_file_not_found_returns_500_html(self):
        with patch(
            'review.views.load_model',
            side_effect=FileNotFoundError('/secret/path/model.keras'),
        ):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response['Content-Type'].split(';')[0], 'text/html')
        self.assertTemplateUsed(response, 'review_analysis_error.html')
        self.assertContains(response, 'model files are missing', status_code=500)
        self.assertNotContains(response, '/secret/path', status_code=500)
        self.assertNotContains(response, 'model.keras', status_code=500)

    def test_generic_exception_returns_500_html_without_leak(self):
        with mock_sentiment(
            predict_side_effect=RuntimeError('secret inference internals')
        ):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response['Content-Type'].split(';')[0], 'text/html')
        self.assertTemplateUsed(response, 'review_analysis_error.html')
        self.assertContains(response, 'failed unexpectedly', status_code=500)
        self.assertNotContains(response, 'secret inference internals', status_code=500)
        self.assertNotContains(response, 'Traceback', status_code=500)

    def test_artifact_mismatch_returns_safe_500_html(self):
        from review.ml_artifacts import ArtifactIntegrityError

        with patch(
            'review.views.load_model',
            side_effect=ArtifactIntegrityError('checksum mismatch /secret'),
        ):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response['Content-Type'].split(';')[0], 'text/html')
        self.assertTemplateUsed(response, 'review_analysis_error.html')
        self.assertContains(response, 'missing or mismatched', status_code=500)
        self.assertNotContains(response, 'checksum', status_code=500)
        self.assertNotContains(response, '/secret', status_code=500)

    def test_success_response_is_html(self):
        with mock_sentiment(predict_return=[[0.8]]):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'].split(';')[0], 'text/html')
        self.assertTemplateUsed(response, 'review_analysis_result.html')

    def test_unauthenticated_is_redirect_not_json(self):
        self.client.logout()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)
        self.assertNotEqual(response.get('Content-Type', ''), 'application/json')


class ReviewAnalyseLoggingTests(TestCase):
    """Inference logging uses review_id + timing; never review text or secrets."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='loguser', password='SecretPass999!')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.private_phrase = 'UNIQUE_PRIVATE_REVIEW_TEXT_DO_NOT_LOG'
        self.review = Review.objects.create(
            user=self.user,
            text=self.private_phrase,
            movie_name='Logging Movie',
            photo=photo,
        )
        self.url = reverse('review_analyse', args=[self.review.pk])
        self.client.login(username='loguser', password='SecretPass999!')

    def _joined_logs(self, capture):
        return '\n'.join(capture.output)

    def test_successful_analysis_logs_inference_timing(self):
        with self.assertLogs('review.views', level='INFO') as captured:
            with mock_sentiment(predict_return=[[0.88]]):
                response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        joined = self._joined_logs(captured)
        self.assertIn(f'review_id={self.review.pk}', joined)
        self.assertIn('Sentiment inference ok', joined)
        self.assertIn('sentiment=positive', joined)
        self.assertIn('predict_ms=', joined)
        self.assertNotIn(self.private_phrase, joined)
        self.assertNotIn('SecretPass999!', joined)
        session_key = self.client.session.session_key
        if session_key:
            self.assertNotIn(session_key, joined)

    def test_cache_hit_logs_without_inference(self):
        with mock_sentiment(predict_return=[[0.9]]):
            self.assertEqual(self.client.get(self.url).status_code, 200)
        with self.assertLogs('review.views', level='INFO') as captured:
            with mock_sentiment(predict_return=[[0.1]]):
                response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'positive')
        joined = self._joined_logs(captured)
        self.assertIn('Sentiment cache hit', joined)
        self.assertIn(f'review_id={self.review.pk}', joined)
        self.assertNotIn('Sentiment inference ok', joined)
        self.assertNotIn(self.private_phrase, joined)

    def test_inference_exception_logs_error_without_secrets(self):
        with self.assertLogs('review.views', level='ERROR') as captured:
            with mock_sentiment(
                predict_side_effect=RuntimeError('secret inference internals')
            ):
                response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        joined = self._joined_logs(captured)
        self.assertIn('Unexpected sentiment analysis failure', joined)
        self.assertIn(f'review_id={self.review.pk}', joined)
        self.assertNotIn(self.private_phrase, joined)
        self.assertNotIn('SecretPass999!', joined)
        from django.conf import settings

        self.assertNotIn(settings.SECRET_KEY, joined)

    def test_missing_artifacts_log_omits_filesystem_path(self):
        with self.assertLogs('review.views', level='ERROR') as captured:
            with patch(
                'review.views.load_model',
                side_effect=FileNotFoundError('/secret/path/model.keras'),
            ):
                response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        joined = self._joined_logs(captured)
        self.assertIn('Missing sentiment artifacts', joined)
        self.assertIn(f'review_id={self.review.pk}', joined)
        self.assertNotIn('/secret/path', joined)
        self.assertNotIn('model.keras', joined)
        self.assertNotIn(self.private_phrase, joined)

    def test_artifact_integrity_log_omits_exception_detail(self):
        from review.ml_artifacts import ArtifactIntegrityError

        with self.assertLogs('review.views', level='ERROR') as captured:
            with patch(
                'review.views.load_model',
                side_effect=ArtifactIntegrityError('checksum mismatch /secret'),
            ):
                response = self.client.get(self.url)
        self.assertEqual(response.status_code, 500)
        joined = self._joined_logs(captured)
        self.assertIn('Sentiment artifact integrity error', joined)
        self.assertIn(f'review_id={self.review.pk}', joined)
        self.assertNotIn('checksum', joined)
        self.assertNotIn('/secret', joined)


class ReviewEditAuthorizationTests(TestCase):
    """Edit requires login and ownership."""

    def setUp(self):
        self.client = Client()
        self.owner = User.objects.create_user(username='owner', password='pass12345')
        self.other = User.objects.create_user(username='other', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.owner,
            text='Original review text.',
            movie_name='Edit Movie',
            photo=photo,
        )
        self.url = reverse('review_edit', args=[self.review.pk])
        self.missing_url = reverse('review_edit', args=[999999])

    def test_owner_can_view_edit_form(self):
        self.client.login(username='owner', password='pass12345')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_form.html')

    def test_owner_can_edit_own_review(self):
        self.client.login(username='owner', password='pass12345')
        response = self.client.post(
            self.url,
            {
                'text': 'Updated review text.',
                'movie_name': 'Edit Movie',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('review_list'))
        self.review.refresh_from_db()
        self.assertEqual(self.review.text, 'Updated review text.')
        self.assertEqual(self.review.user_id, self.owner.id)

    def test_other_user_cannot_edit_review(self):
        self.client.login(username='other', password='pass12345')
        response = self.client.post(
            self.url,
            {
                'text': 'Hijacked text.',
                'movie_name': 'Edit Movie',
            },
        )
        self.assertEqual(response.status_code, 404)
        self.review.refresh_from_db()
        self.assertEqual(self.review.text, 'Original review text.')

    def test_other_user_cannot_view_edit_form(self):
        self.client.login(username='other', password='pass12345')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 404)

    def test_unauthenticated_user_cannot_edit(self):
        response = self.client.post(
            self.url,
            {'text': 'Anon edit.', 'movie_name': 'Edit Movie'},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)
        self.review.refresh_from_db()
        self.assertEqual(self.review.text, 'Original review text.')

    def test_unauthenticated_user_cannot_view_edit_form(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_nonexistent_review_returns_404(self):
        self.client.login(username='owner', password='pass12345')
        response = self.client.get(self.missing_url)
        self.assertEqual(response.status_code, 404)

    def test_edit_post_requires_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.login(username='owner', password='pass12345')
        response = csrf_client.post(
            self.url,
            {'text': 'No CSRF.', 'movie_name': 'Edit Movie'},
        )
        self.assertEqual(response.status_code, 403)
        self.review.refresh_from_db()
        self.assertEqual(self.review.text, 'Original review text.')


class ReviewDeleteAuthorizationTests(TestCase):
    """Delete requires login and ownership (same pattern as review_edit)."""

    def setUp(self):
        self.client = Client()
        self.owner = User.objects.create_user(username='owner', password='pass12345')
        self.other = User.objects.create_user(username='other', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.owner,
            text='A review that may be deleted.',
            movie_name='Delete Movie',
            photo=photo,
        )
        self.url = reverse('review_delete', args=[self.review.pk])
        self.missing_url = reverse('review_delete', args=[999999])

    def test_owner_can_view_delete_confirmation(self):
        self.client.login(username='owner', password='pass12345')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_confirm_delete.html')

    def test_owner_can_delete_own_review(self):
        self.client.login(username='owner', password='pass12345')
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('review_list'))
        self.assertFalse(Review.objects.filter(pk=self.review.pk).exists())

    def test_other_user_cannot_delete_review(self):
        self.client.login(username='other', password='pass12345')
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 404)
        self.assertTrue(Review.objects.filter(pk=self.review.pk).exists())

    def test_other_user_cannot_view_delete_confirmation(self):
        self.client.login(username='other', password='pass12345')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 404)

    def test_unauthenticated_user_cannot_delete(self):
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)
        self.assertTrue(Review.objects.filter(pk=self.review.pk).exists())

    def test_unauthenticated_user_cannot_view_delete_confirmation(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_nonexistent_review_returns_404(self):
        self.client.login(username='owner', password='pass12345')
        response = self.client.get(self.missing_url)
        self.assertEqual(response.status_code, 404)

    def test_delete_post_requires_csrf(self):
        self.client.login(username='owner', password='pass12345')
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.login(username='owner', password='pass12345')
        response = csrf_client.post(self.url)
        self.assertEqual(response.status_code, 403)
        self.assertTrue(Review.objects.filter(pk=self.review.pk).exists())


class ProtectedViewsAuthenticationTests(TestCase):
    """Session-auth gate for protected review endpoints."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='authuser', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.user,
            text='Protected ops review.',
            movie_name='Auth Movie',
            photo=photo,
        )

    def test_anonymous_cannot_access_list_create_edit_delete_analyse(self):
        urls = [
            reverse('review_list'),
            reverse('review_create'),
            reverse('review_edit', args=[self.review.pk]),
            reverse('review_delete', args=[self.review.pk]),
            reverse('review_analyse', args=[self.review.pk]),
        ]
        for url in urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302, msg=url)
            self.assertIn('/accounts/login/', response.url, msg=url)

    def test_create_post_requires_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.login(username='authuser', password='pass12345')
        response = csrf_client.post(
            reverse('review_create'),
            {'text': 'CSRF blocked.', 'movie_name': 'Auth Movie'},
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(
            Review.objects.filter(user=self.user, text='CSRF blocked.').exists()
        )


class UserRegistrationTests(TestCase):
    """Registration relies on UserCreationForm.save() for password hashing."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('register')
        self.valid_payload = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'ComplexPass123!',
            'password2': 'ComplexPass123!',
        }

    def test_register_hashes_password_and_allows_login(self):
        response = self.client.post(self.url, self.valid_payload)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('review_list'))

        user = User.objects.get(username='newuser')
        self.assertNotEqual(user.password, self.valid_payload['password1'])
        self.assertTrue(user.password.startswith(('pbkdf2_', 'argon2', 'bcrypt', 'scrypt')))
        self.assertTrue(user.check_password(self.valid_payload['password1']))

        self.client.logout()
        self.assertTrue(
            self.client.login(
                username='newuser',
                password=self.valid_payload['password1'],
            )
        )

    def test_register_rejects_mismatched_passwords(self):
        payload = {
            **self.valid_payload,
            'password2': 'DifferentPass123!',
        }
        response = self.client.post(self.url, payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='newuser').exists())
        self.assertTrue(response.context['form'].errors)

    def test_register_rejects_too_short_password(self):
        payload = {
            **self.valid_payload,
            'password1': 'ab',
            'password2': 'ab',
        }
        response = self.client.post(self.url, payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='newuser').exists())
        self.assertIn('password2', response.context['form'].errors)

    def test_register_rejects_common_password(self):
        payload = {
            **self.valid_payload,
            'password1': 'password',
            'password2': 'password',
        }
        response = self.client.post(self.url, payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='newuser').exists())
        self.assertTrue(response.context['form'].errors)

    def test_register_post_requires_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        response = csrf_client.post(self.url, self.valid_payload)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(User.objects.filter(username='newuser').exists())


class ReviewTextValidationTests(TestCase):
    """Empty/whitespace-only review text must be rejected at the form boundary."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='reviewer', password='pass12345')
        self.create_url = reverse('review_create')
        self.client.login(username='reviewer', password='pass12345')

    def _post_review(self, text, movie_name='Test Movie'):
        return self.client.post(
            self.create_url,
            {'text': text, 'movie_name': movie_name},
        )

    def test_empty_text_rejected(self):
        response = self._post_review('')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Review.objects.filter(user=self.user).exists())
        self.assertIn('text', response.context['form'].errors)

    def test_whitespace_only_text_rejected(self):
        response = self._post_review('   \t\n  ')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Review.objects.filter(user=self.user).exists())
        self.assertIn('text', response.context['form'].errors)

    def test_normal_text_accepted(self):
        response = self._post_review('A thoughtful and exciting film.')
        self.assertEqual(response.status_code, 302)
        review = Review.objects.get(user=self.user)
        self.assertEqual(review.text, 'A thoughtful and exciting film.')

    def test_leading_trailing_whitespace_normalized(self):
        response = self._post_review('  Still a valid review.  ')
        self.assertEqual(response.status_code, 302)
        review = Review.objects.get(user=self.user)
        self.assertEqual(review.text, 'Still a valid review.')

    def test_form_clean_text_unit(self):
        from .forms import ReviewForm

        form = ReviewForm(data={'text': '   ', 'movie_name': 'X'})
        self.assertFalse(form.is_valid())
        self.assertIn('text', form.errors)

        form = ReviewForm(data={'text': 'Great pacing.', 'movie_name': 'X'})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data['text'], 'Great pacing.')


class ReviewAnalyseRateLimitTests(TestCase):
    """ML inference is limited per authenticated user (django-ratelimit key='user')."""

    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        self.client = Client()
        self.user_a = User.objects.create_user(username='usera', password='pass12345')
        self.user_b = User.objects.create_user(username='userb', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.user_a,
            text='Rate limit analysis review text.',
            movie_name='Rate Limit Movie',
            photo=photo,
        )
        self.url = reverse('review_analyse', args=[self.review.pk])

    def test_requests_below_limit_succeed(self):
        self.client.login(username='usera', password='pass12345')
        with self.settings(REVIEW_ANALYSE_RATELIMIT='3/h'), mock_sentiment():
            for _ in range(3):
                response = self.client.get(self.url)
                self.assertEqual(response.status_code, 200)

    def test_requests_over_limit_are_blocked(self):
        self.client.login(username='usera', password='pass12345')
        with self.settings(REVIEW_ANALYSE_RATELIMIT='2/h'), mock_sentiment():
            self.assertEqual(self.client.get(self.url).status_code, 200)
            self.assertEqual(self.client.get(self.url).status_code, 200)
            limited = self.client.get(self.url)
            self.assertEqual(limited.status_code, 403)
            # Generic denial; do not leak internals.
            self.assertNotIn(b'traceback', limited.content.lower())
            self.assertNotIn(b'SECRET', limited.content)

    def test_unauthenticated_still_redirected_to_login(self):
        with self.settings(REVIEW_ANALYSE_RATELIMIT='1/h'):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_rate_limit_is_per_user(self):
        with self.settings(REVIEW_ANALYSE_RATELIMIT='1/h'), mock_sentiment():
            self.client.login(username='usera', password='pass12345')
            self.assertEqual(self.client.get(self.url).status_code, 200)
            self.assertEqual(self.client.get(self.url).status_code, 403)

            self.client.logout()
            self.client.login(username='userb', password='pass12345')
            # Separate user key bucket — not blocked by usera's limit.
            self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_cannot_abuse_endpoint_indefinitely(self):
        self.client.login(username='usera', password='pass12345')
        with self.settings(REVIEW_ANALYSE_RATELIMIT='5/h'), mock_sentiment():
            statuses = [self.client.get(self.url).status_code for _ in range(8)]
        self.assertEqual(statuses.count(200), 5)
        self.assertEqual(statuses.count(403), 3)
        self.assertTrue(all(s in (200, 403) for s in statuses))


class LoginRateLimitTests(TestCase):
    """Built-in LoginView subclass throttles POST by IP and username."""

    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        self.client = Client()
        self.user = User.objects.create_user(username='loginuser', password='CorrectPass123!')
        self.url = reverse('login')

    def test_login_succeeds_under_limit(self):
        with self.settings(LOGIN_RATELIMIT='5/m'):
            response = self.client.post(
                self.url,
                {'username': 'loginuser', 'password': 'CorrectPass123!'},
            )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            response.wsgi_request.user.is_authenticated
            or '_auth_user_id' in self.client.session
        )

    def test_login_throttled_after_excess_posts(self):
        with self.settings(LOGIN_RATELIMIT='3/m'):
            for _ in range(3):
                response = self.client.post(
                    self.url,
                    {'username': 'loginuser', 'password': 'wrong-password'},
                )
                self.assertIn(response.status_code, (200, 302))
            limited = self.client.post(
                self.url,
                {'username': 'loginuser', 'password': 'CorrectPass123!'},
            )
            self.assertEqual(limited.status_code, 403)
            # Even correct password is blocked once rate-limited (no auth bypass).
            self.assertNotIn('_auth_user_id', self.client.session)

    def test_login_get_form_not_blocked_by_post_limit(self):
        with self.settings(LOGIN_RATELIMIT='1/m'):
            self.client.post(
                self.url,
                {'username': 'loginuser', 'password': 'wrong'},
            )
            # Second POST is limited, but GET for the form remains available.
            self.assertEqual(
                self.client.post(
                    self.url,
                    {'username': 'loginuser', 'password': 'wrong'},
                ).status_code,
                403,
            )
            self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_login_rate_limit_keyed_by_username(self):
        User.objects.create_user(username='otherlogin', password='CorrectPass123!')
        with self.settings(LOGIN_RATELIMIT='1/m'):
            self.assertIn(
                self.client.post(
                    self.url,
                    {'username': 'loginuser', 'password': 'wrong'},
                ).status_code,
                (200, 302),
            )
            self.assertEqual(
                self.client.post(
                    self.url,
                    {'username': 'loginuser', 'password': 'CorrectPass123!'},
                ).status_code,
                403,
            )
            # Different username has its own counter (IP limit is also 1/m —
            # use a fresh client IP via REMOTE_ADDR to isolate username key).
            other_client = Client(REMOTE_ADDR='10.0.0.99')
            response = other_client.post(
                self.url,
                {'username': 'otherlogin', 'password': 'CorrectPass123!'},
            )
            self.assertEqual(response.status_code, 302)
            self.assertIn('_auth_user_id', other_client.session)


class SentimentCacheTests(TestCase):
    """Per-review sentiment cache: infer once, reuse until text changes."""

    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        self.client = Client()
        self.owner = User.objects.create_user(username='cacheowner', password='pass12345')
        self.other = User.objects.create_user(username='cacheother', password='pass12345')
        photo = SimpleUploadedFile('t.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.owner,
            text='Cached sentiment review text.',
            movie_name='Cache Movie',
            photo=photo,
        )
        self.url = reverse('review_analyse', args=[self.review.pk])
        self.client.login(username='cacheowner', password='pass12345')

    def test_first_analysis_performs_inference_and_stores_cache(self):
        with mock_sentiment(predict_return=[[0.88]]):
            from review import views as review_views

            response = self.client.get(self.url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(review_views.model.predict.call_count, 1)

        self.review.refresh_from_db()
        self.assertEqual(self.review.sentiment_label, 'positive')
        self.assertAlmostEqual(self.review.sentiment_score, 0.88)
        self.assertIsNotNone(self.review.sentiment_analyzed_at)
        self.assertEqual(self.review.sentiment_text_hash, self.review.compute_text_hash())
        self.assertContains(response, '88% positive probability')

    def test_repeated_analysis_uses_cached_result(self):
        with mock_sentiment(predict_return=[[0.77]]):
            from review import views as review_views

            first = self.client.get(self.url)
            second = self.client.get(self.url)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(review_views.model.predict.call_count, 1)
            self.assertEqual(second.context['prediction_percent'], 77)
            self.assertEqual(second.context['sentiment'], 'positive')

    def test_editing_text_invalidates_cache_and_recomputes(self):
        with mock_sentiment(predict_return=[[0.9]]):
            self.assertEqual(self.client.get(self.url).status_code, 200)

        self.review.refresh_from_db()
        self.assertEqual(self.review.sentiment_label, 'positive')

        edit_url = reverse('review_edit', args=[self.review.pk])
        response = self.client.post(
            edit_url,
            {
                'text': 'Completely rewritten review after analysis.',
                'movie_name': 'Cache Movie',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.review.refresh_from_db()
        self.assertIsNone(self.review.sentiment_label)
        self.assertIsNone(self.review.sentiment_score)
        self.assertIsNone(self.review.sentiment_text_hash)
        self.assertFalse(self.review.has_valid_sentiment_cache())

        with mock_sentiment(predict_return=[[0.12]]):
            from review import views as review_views

            again = self.client.get(self.url)
            self.assertEqual(again.status_code, 200)
            self.assertEqual(review_views.model.predict.call_count, 1)
            self.assertEqual(again.context['sentiment'], 'negative')
            self.assertEqual(again.context['prediction_percent'], 12)

        self.review.refresh_from_db()
        self.assertEqual(self.review.sentiment_label, 'negative')
        self.assertEqual(
            self.review.sentiment_text_hash,
            self.review.compute_text_hash(),
        )

    def test_cached_results_do_not_cross_reviews(self):
        photo = SimpleUploadedFile('u.png', _TINY_PNG, content_type='image/png')
        other_review = Review.objects.create(
            user=self.owner,
            text='A different review that must not share cache.',
            movie_name='Other Cache Movie',
            photo=photo,
        )
        other_url = reverse('review_analyse', args=[other_review.pk])

        with mock_sentiment(predict_return=[[0.95]]):
            from review import views as review_views

            self.assertEqual(self.client.get(self.url).status_code, 200)
            self.assertEqual(review_views.model.predict.call_count, 1)

        with mock_sentiment(predict_return=[[0.05]]):
            from review import views as review_views

            response = self.client.get(other_url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(review_views.model.predict.call_count, 1)
            self.assertEqual(response.context['sentiment'], 'negative')

        self.review.refresh_from_db()
        other_review.refresh_from_db()
        self.assertEqual(self.review.sentiment_label, 'positive')
        self.assertEqual(other_review.sentiment_label, 'negative')
        self.assertNotEqual(
            self.review.sentiment_text_hash,
            other_review.sentiment_text_hash,
        )

    def test_stale_hash_mismatch_forces_recompute(self):
        """Hash check blocks presenting old sentiment if text changed out-of-band."""
        with mock_sentiment(predict_return=[[0.91]]):
            self.assertEqual(self.client.get(self.url).status_code, 200)

        Review.objects.filter(pk=self.review.pk).update(
            text='Text changed without going through review_edit.',
        )
        self.review.refresh_from_db()
        self.assertFalse(self.review.has_valid_sentiment_cache())

        with mock_sentiment(predict_return=[[0.2]]):
            from review import views as review_views

            response = self.client.get(self.url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(review_views.model.predict.call_count, 1)
            self.assertEqual(response.context['sentiment'], 'negative')

    def test_analyse_still_requires_authentication(self):
        self.client.logout()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)


class MlArtifactPairingTests(TestCase):
    """model_version.json ties model + tokenizer as one deployable unit."""

    def setUp(self):
        import tempfile

        from review.ml_artifacts import artifact_paths, write_version_manifest

        self._tmpdir = tempfile.TemporaryDirectory()
        self.artifact_dir = self._tmpdir.name
        self.paths = artifact_paths(self.artifact_dir)
        self._write_version_manifest = write_version_manifest

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_files(self, model_bytes=b'model-bytes-v1', tokenizer_text='{"tok":1}'):
        with open(self.paths['model'], 'wb') as handle:
            handle.write(model_bytes)
        with open(self.paths['tokenizer'], 'w', encoding='utf-8') as handle:
            handle.write(tokenizer_text)

    def test_correct_pair_verifies_successfully(self):
        from review.ml_artifacts import build_version_manifest, verify_artifact_pair

        self._write_files()
        manifest = build_version_manifest(self.artifact_dir, tensorflow_version='test')
        self._write_version_manifest(self.artifact_dir, manifest)
        verified = verify_artifact_pair(self.artifact_dir)
        self.assertEqual(verified['version_id'], manifest['version_id'])
        self.assertEqual(
            verified['artifacts']['model']['sha256'],
            manifest['artifacts']['model']['sha256'],
        )

    def test_missing_model_is_detected(self):
        from review.ml_artifacts import build_version_manifest, verify_artifact_pair

        self._write_files()
        manifest = build_version_manifest(self.artifact_dir, tensorflow_version='test')
        self._write_version_manifest(self.artifact_dir, manifest)
        os.remove(self.paths['model'])
        with self.assertRaises(FileNotFoundError):
            verify_artifact_pair(self.artifact_dir)

    def test_missing_tokenizer_is_detected(self):
        from review.ml_artifacts import build_version_manifest, verify_artifact_pair

        self._write_files()
        manifest = build_version_manifest(self.artifact_dir, tensorflow_version='test')
        self._write_version_manifest(self.artifact_dir, manifest)
        os.remove(self.paths['tokenizer'])
        with self.assertRaises(FileNotFoundError):
            verify_artifact_pair(self.artifact_dir)

    def test_mismatched_artifacts_are_detected(self):
        from review.ml_artifacts import (
            ArtifactIntegrityError,
            build_version_manifest,
            verify_artifact_pair,
        )

        self._write_files(model_bytes=b'original-model')
        manifest = build_version_manifest(self.artifact_dir, tensorflow_version='test')
        self._write_version_manifest(self.artifact_dir, manifest)
        # Swap in a different model file while keeping the old checksum.
        with open(self.paths['model'], 'wb') as handle:
            handle.write(b'tampered-model-bytes')
        with self.assertRaises(ArtifactIntegrityError) as ctx:
            verify_artifact_pair(self.artifact_dir)
        self.assertIn('checksum mismatch', str(ctx.exception))

    def test_missing_manifest_is_detected(self):
        from review.ml_artifacts import ArtifactIntegrityError, verify_artifact_pair

        self._write_files()
        with self.assertRaises(ArtifactIntegrityError) as ctx:
            verify_artifact_pair(self.artifact_dir)
        self.assertIn('model_version.json', str(ctx.exception))


class DeployedKerasArtifactIntegrationTests(TestCase):
    """Loads real on-disk paired artifacts when present (skipped in TF-less CI)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from django.conf import settings

        from review.ml_artifacts import MODEL_FILENAME, TOKENIZER_FILENAME, VERSION_FILENAME

        cls.artifact_dir = os.path.join(settings.BASE_DIR, 'review', 'models')
        cls.required = [
            os.path.join(cls.artifact_dir, name)
            for name in (MODEL_FILENAME, TOKENIZER_FILENAME, VERSION_FILENAME)
        ]

    def _artifacts_ready(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            return False
        return all(os.path.isfile(path) for path in self.required)

    def test_correct_pair_loads_and_runs_inference(self):
        if not self._artifacts_ready():
            self.skipTest('Paired Keras artifacts or TensorFlow not available')

        from review.ml_artifacts import load_paired_artifacts

        model, tokenizer, pad_sequences, manifest = load_paired_artifacts(self.artifact_dir)
        self.assertTrue(manifest.get('version_id'))
        sequence = tokenizer.texts_to_sequences(['This movie was fantastic and thrilling.'])
        padded = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded, verbose=0)
        self.assertEqual(prediction.shape[-1], 1)
        score = float(prediction[0][0])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class ReviewListPaginationTests(TestCase):
    """review_list uses Django Paginator (page size 12) and preserves search."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='pager', password='pass12345')
        self.client.login(username='pager', password='pass12345')
        self.url = reverse('review_list')

    def _make_reviews(self, count, movie_name='Page Movie', text_prefix='Review'):
        created = []
        for i in range(count):
            photo = SimpleUploadedFile(
                f't{i}.png',
                _TINY_PNG,
                content_type='image/png',
            )
            created.append(
                Review.objects.create(
                    user=self.user,
                    text=f'{text_prefix} {i}',
                    movie_name=movie_name,
                    photo=photo,
                )
            )
        return created

    def test_first_page(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        self._make_reviews(REVIEW_LIST_PAGE_SIZE + 3)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_list.html')
        page_obj = response.context['page_obj']
        self.assertEqual(page_obj.number, 1)
        self.assertEqual(len(page_obj.object_list), REVIEW_LIST_PAGE_SIZE)
        self.assertTrue(page_obj.has_next())
        self.assertContains(response, 'Page 1 of')
        self.assertContains(response, 'Next')

    def test_second_page(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        created = self._make_reviews(REVIEW_LIST_PAGE_SIZE + 5)
        response = self.client.get(self.url, {'page': 2})
        self.assertEqual(response.status_code, 200)
        page_obj = response.context['page_obj']
        self.assertEqual(page_obj.number, 2)
        self.assertEqual(len(page_obj.object_list), 5)
        self.assertTrue(page_obj.has_previous())
        self.assertContains(response, 'Previous')
        page_ids = [r.pk for r in page_obj.object_list]
        self.assertEqual(page_ids, [r.pk for r in created[REVIEW_LIST_PAGE_SIZE:]])

    def test_page_boundary_exact_page_size(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        self._make_reviews(REVIEW_LIST_PAGE_SIZE)
        response = self.client.get(self.url)
        page_obj = response.context['page_obj']
        self.assertEqual(page_obj.paginator.num_pages, 1)
        self.assertFalse(page_obj.has_next())
        self.assertEqual(len(page_obj.object_list), REVIEW_LIST_PAGE_SIZE)

    def test_invalid_page_uses_django_get_page_behavior(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        self._make_reviews(REVIEW_LIST_PAGE_SIZE + 1)
        response = self.client.get(self.url, {'page': 'abc'})
        self.assertEqual(response.context['page_obj'].number, 1)
        response = self.client.get(self.url, {'page': 999})
        self.assertEqual(response.context['page_obj'].number, 2)

    def test_search_with_pagination(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        self._make_reviews(REVIEW_LIST_PAGE_SIZE + 2, movie_name='Inception')
        response = self.client.get(self.url, {'q': 'Inception', 'page': 1})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_search.html')
        page_obj = response.context['page_obj']
        self.assertEqual(page_obj.number, 1)
        self.assertTrue(page_obj.has_next())
        self.assertEqual(len(page_obj.object_list), REVIEW_LIST_PAGE_SIZE)

    def test_search_parameters_preserved_in_pagination_links(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        self._make_reviews(REVIEW_LIST_PAGE_SIZE + 1, movie_name='Matrix')
        response = self.client.get(self.url, {'q': 'Matrix', 'page': 1})
        self.assertContains(response, 'q=Matrix')
        self.assertContains(response, 'href="?page=2&amp;q=Matrix"')

    def test_empty_result_set(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        page_obj = response.context['page_obj']
        self.assertEqual(page_obj.paginator.count, 0)
        self.assertEqual(len(page_obj.object_list), 0)

        response = self.client.get(self.url, {'q': 'NoSuchMovieXYZ'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_search.html')
        self.assertContains(response, 'No results found')
        self.assertEqual(response.context['page_obj'].paginator.count, 0)

    def test_list_query_is_paginated_not_full_scan_into_context(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        self._make_reviews(REVIEW_LIST_PAGE_SIZE + 8)
        response = self.client.get(self.url)
        page_obj = response.context['page_obj']
        self.assertLessEqual(len(list(page_obj.object_list)), REVIEW_LIST_PAGE_SIZE)
        self.assertEqual(page_obj.paginator.count, REVIEW_LIST_PAGE_SIZE + 8)


class ReviewSearchBehaviorTests(TestCase):
    """Search stays case-insensitive on SQLite; query is normalized safely."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='searcher', password='pass12345')
        self.client.login(username='searcher', password='pass12345')
        self.url = reverse('review_list')
        photo = SimpleUploadedFile('s.png', _TINY_PNG, content_type='image/png')
        self.review = Review.objects.create(
            user=self.user,
            text='A searchable review.',
            movie_name='The Dark Knight',
            photo=photo,
        )

    def test_case_insensitive_substring_match(self):
        for needle in ('dark', 'DARK', 'Dark', 'knight', 'KNIGHT'):
            response = self.client.get(self.url, {'q': needle})
            self.assertEqual(response.status_code, 200, msg=needle)
            self.assertTemplateUsed(response, 'review_search.html')
            self.assertContains(response, 'The Dark Knight')
            self.assertEqual(response.context['page_obj'].paginator.count, 1)

    def test_whitespace_only_query_lists_all_not_search_template(self):
        response = self.client.get(self.url, {'q': '   '})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_list.html')
        self.assertEqual(response.context['query'], '')
        self.assertEqual(response.context['page_obj'].paginator.count, 1)

    def test_search_still_paginates(self):
        from review.views import REVIEW_LIST_PAGE_SIZE

        for i in range(REVIEW_LIST_PAGE_SIZE + 1):
            Review.objects.create(
                user=self.user,
                text=f'Search page {i}',
                movie_name='Batman Begins',
                photo=SimpleUploadedFile(f'b{i}.png', _TINY_PNG, content_type='image/png'),
            )
        response = self.client.get(self.url, {'q': 'batman', 'page': 2})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_search.html')
        self.assertEqual(response.context['page_obj'].number, 2)
        self.assertContains(response, 'q=batman')

    def test_normalize_review_search_query_unit(self):
        from review.views import (
            REVIEW_SEARCH_QUERY_MAX_LEN,
            normalize_review_search_query,
        )

        self.assertEqual(normalize_review_search_query(None), '')
        self.assertEqual(normalize_review_search_query('  Inception  '), 'Inception')
        long = 'x' * (REVIEW_SEARCH_QUERY_MAX_LEN + 50)
        self.assertEqual(
            len(normalize_review_search_query(long)),
            REVIEW_SEARCH_QUERY_MAX_LEN,
        )


class ReviewPhotoOptimizationTests(TestCase):
    """Review.photo is resized on upload only; text-only saves leave it alone."""

    def setUp(self):
        self.user = User.objects.create_user(username='photouser', password='pass12345')
        self.client = Client()
        self.client.login(username='photouser', password='pass12345')

    def _rgb_upload(self, width, height, name='shot.jpg', fmt='JPEG'):
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        image = Image.new('RGB', (width, height), color=(20, 120, 200))
        save_kwargs = {'format': fmt}
        if fmt == 'JPEG':
            save_kwargs['quality'] = 95
        image.save(buffer, **save_kwargs)
        buffer.seek(0)
        content_type = 'image/jpeg' if fmt == 'JPEG' else f'image/{fmt.lower()}'
        return SimpleUploadedFile(name, buffer.read(), content_type=content_type)

    def _png_rgba_upload(self, width, height, name='alpha.png'):
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        image = Image.new('RGBA', (width, height), color=(255, 0, 0, 128))
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return SimpleUploadedFile(name, buffer.read(), content_type='image/png')

    def _open_stored(self, review):
        from PIL import Image

        review.photo.open('rb')
        try:
            with Image.open(review.photo) as image:
                image.load()
                return image.copy()
        finally:
            review.photo.close()

    def test_upload_normal_image_keeps_dimensions(self):
        from review.image_utils import PHOTO_MAX_EDGE

        upload = self._rgb_upload(640, 480, name='normal.jpg')
        review = Review.objects.create(
            user=self.user,
            text='Normal sized photo review.',
            movie_name='Normal',
            photo=upload,
        )
        stored = self._open_stored(review)
        self.assertEqual(stored.size, (640, 480))
        self.assertLessEqual(max(stored.size), PHOTO_MAX_EDGE)

    def test_upload_oversized_image_is_resized_preserving_aspect(self):
        from review.image_utils import PHOTO_MAX_EDGE

        upload = self._rgb_upload(4000, 2000, name='huge.jpg')
        review = Review.objects.create(
            user=self.user,
            text='Oversized photo review.',
            movie_name='Huge',
            photo=upload,
        )
        stored = self._open_stored(review)
        width, height = stored.size
        self.assertEqual(max(width, height), PHOTO_MAX_EDGE)
        self.assertEqual(width, PHOTO_MAX_EDGE)
        self.assertEqual(height, PHOTO_MAX_EDGE // 2)

    def test_upload_exif_orientation_is_applied_before_resize(self):
        from io import BytesIO

        from PIL import Image
        from review.image_utils import PHOTO_MAX_EDGE

        # Wide image tagged as Orientation=6 (rotate 90 CW) → tall after transpose.
        buffer = BytesIO()
        image = Image.new('RGB', (2000, 1000), color=(10, 20, 30))
        exif = image.getexif()
        exif[274] = 6
        image.save(buffer, format='JPEG', quality=95, exif=exif)
        buffer.seek(0)
        upload = SimpleUploadedFile(
            'oriented.jpg', buffer.read(), content_type='image/jpeg'
        )
        review = Review.objects.create(
            user=self.user,
            text='Oriented photo review.',
            movie_name='Orient',
            photo=upload,
        )
        stored = self._open_stored(review)
        width, height = stored.size
        self.assertEqual(max(width, height), PHOTO_MAX_EDGE)
        # After orientation, portrait (1000x2000) scales to 600x1200.
        self.assertEqual(width, PHOTO_MAX_EDGE // 2)
        self.assertEqual(height, PHOTO_MAX_EDGE)

    def test_update_without_replacing_photo_does_not_reprocess(self):
        upload = self._rgb_upload(800, 600, name='keep.jpg')
        review = Review.objects.create(
            user=self.user,
            text='Original text.',
            movie_name='Keep',
            photo=upload,
        )
        original_name = review.photo.name
        with patch('review.models.optimize_review_photo') as optimize_mock:
            review.text = 'Updated text only.'
            review.save()
            optimize_mock.assert_not_called()
        review.refresh_from_db()
        self.assertEqual(review.photo.name, original_name)
        self.assertEqual(review.text, 'Updated text only.')
        stored = self._open_stored(review)
        self.assertEqual(stored.size, (800, 600))

    def test_sentiment_cache_save_skips_photo_processing(self):
        upload = self._rgb_upload(500, 500, name='cache.jpg')
        review = Review.objects.create(
            user=self.user,
            text='Cache photo review.',
            movie_name='Cache',
            photo=upload,
        )
        with patch('review.models.optimize_review_photo') as optimize_mock:
            review.store_sentiment_cache('positive', 0.91)
            optimize_mock.assert_not_called()

    def test_invalid_image_rejected_by_form(self):
        from review.forms import ReviewForm

        form = ReviewForm(
            data={'text': 'Bad photo', 'movie_name': 'Bad'},
            files={
                'photo': SimpleUploadedFile(
                    'nope.jpg', b'not-an-image', content_type='image/jpeg'
                )
            },
        )
        self.assertFalse(form.is_valid())
        self.assertIn('photo', form.errors)

    def test_png_transparency_preserved_when_resized(self):
        from review.image_utils import PHOTO_MAX_EDGE

        upload = self._png_rgba_upload(1800, 1800, name='glass.png')
        review = Review.objects.create(
            user=self.user,
            text='Transparent PNG review.',
            movie_name='PNG',
            photo=upload,
        )
        stored = self._open_stored(review)
        self.assertTrue(review.photo.name.lower().endswith('.png'))
        self.assertEqual(stored.mode, 'RGBA')
        self.assertEqual(max(stored.size), PHOTO_MAX_EDGE)
        pixel = stored.getpixel((0, 0))
        self.assertEqual(len(pixel), 4)
        self.assertLess(pixel[3], 255)

    def test_create_via_form_upload_is_displayable(self):
        from review.image_utils import PHOTO_MAX_EDGE

        url = reverse('review_create')
        response = self.client.post(
            url,
            {
                'text': 'Form upload review.',
                'movie_name': 'FormMovie',
                'photo': self._rgb_upload(2500, 1000, name='form.jpg'),
            },
        )
        self.assertEqual(response.status_code, 302)
        review = Review.objects.get(user=self.user, movie_name='FormMovie')
        stored = self._open_stored(review)
        self.assertEqual(max(stored.size), PHOTO_MAX_EDGE)
        self.assertTrue(review.photo.url)
        self.assertTrue(review.photo.storage.exists(review.photo.name))
