# CineSense

## Setup (secrets)

Django requires `DJANGO_SECRET_KEY` in every environment. It is never committed.

1. From the `cinesense` app directory:

```bash
python bootstrap_env.py
```

Or copy the example and set the key yourself:

```bash
cp .env.example .env
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Put the generated value in `.env` as `DJANGO_SECRET_KEY=...`.

For local development, set `DJANGO_DEBUG=True` in `.env` (included when using `bootstrap_env.py`). If `DJANGO_DEBUG` is unset, Django runs with `DEBUG=False`.

`DJANGO_ALLOWED_HOSTS` is a comma-separated list (default `localhost,127.0.0.1` when unset).

2. Run the app as usual (`python manage.py runserver`, etc.).

## Deployment

Set a **new** `DJANGO_SECRET_KEY` (different from any key that ever appeared in git) as an environment variable on every host/CI/platform secret store. Do not reuse a leaked or example value.

Leave `DJANGO_DEBUG` unset or set `DJANGO_DEBUG=False` in production. Never enable debug on public hosts.

Set `DJANGO_ALLOWED_HOSTS` to your real domain(s), e.g. `example.com,www.example.com`. Do not use `*`.

After rotating the key:

- Existing signed cookies and sessions become untrusted automatically.
- Clear server-side sessions if you use the database/cache session backend, e.g. `python manage.py clearsessions` or delete rows in `django_session`.

## Compromised key / git history

If a secret key was committed publicly, treat it as permanently compromised: rotate everywhere, then scrub history (for example `git filter-repo --replace-text`) and force-push only after coordinating with collaborators. GitHub may still retain the old blobs until support/cache expiry; rotation remains mandatory.
