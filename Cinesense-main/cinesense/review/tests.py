from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.urls import reverse

from .models import Review


# Minimal valid 1x1 PNG so analysis templates can resolve review.photo.url
_TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
    b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
    b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
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

    def _mock_sentiment(self):
        tokenizer = MagicMock()
        tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
        model = MagicMock()
        model.predict.return_value = [[0.9]]
        return patch.multiple('review.views', tokenizer=tokenizer, model=model)

    def test_owner_can_analyse_own_review(self):
        self.client.login(username='owner', password='pass12345')
        with self._mock_sentiment():
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'positive')

    def test_other_authenticated_user_can_analyse_visible_review(self):
        # Matches UI: Analyze is offered for every review to any logged-in user.
        self.client.login(username='other', password='pass12345')
        with self._mock_sentiment():
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.review.text)

    def test_unauthenticated_user_is_redirected_to_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_nonexistent_review_returns_404(self):
        self.client.login(username='owner', password='pass12345')
        with self._mock_sentiment():
            response = self.client.get(self.missing_url)
        self.assertEqual(response.status_code, 404)

    def test_post_method_not_allowed(self):
        self.client.login(username='owner', password='pass12345')
        with self._mock_sentiment():
            response = self.client.post(self.url)
        self.assertEqual(response.status_code, 405)

    def test_no_private_review_flag_on_model(self):
        field_names = {f.name for f in Review._meta.get_fields()}
        self.assertNotIn('is_private', field_names)
        self.assertNotIn('visibility', field_names)
        self.assertNotIn('is_public', field_names)


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
