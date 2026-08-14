# CineSense

## Dependencies

Python **3.11–3.12** (CI uses 3.12). From the `Cinesense-main` package directory:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

- `requirements.txt` — runtime (Django app + TensorFlow CPU for sentiment inference)
- `requirements-train.txt` — runtime plus training/notebook deps (`pandas`, `scikit-learn`) for `main.ipynb` / `retrain_oov_artifacts.py`
- `requirements-ci.txt` — lean Django + flake8 stack used by GitHub Actions (TensorFlow omitted; ML is mocked)

Pinned TensorFlow **2.19.1** matches `cinesense/review/models/model_version.json`. NumPy is pulled in transitively by TensorFlow (constrained to `>=1.26,<2.2`).

## Setup (secrets)

Django requires `DJANGO_SECRET_KEY` in every environment. It is never committed.

1. From the `cinesense` app directory (`Cinesense-main/cinesense`):

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

## Database and search scalability

**Current database:** SQLite (`django.db.backends.sqlite3`, file `db.sqlite3`). This is what local development, CI, and the checked-in settings use.

**Production database:** Not configured separately. There is no PostgreSQL (or other) engine in settings/requirements today; SQLite is the intended deployment database unless/until you deliberately migrate.

**Review search:** `review_list` filters with `movie_name__icontains`. On SQLite that becomes a case-insensitive `LIKE '%query%'` and generally **cannot use a normal B-tree index** (leading wildcard). Pagination (12 per page) limits how much is rendered, but the matching scan still grows with the `Review` table.

**What we do not do on SQLite:** add a plain `db_index` / migration that only pretends to fix `icontains`, or introduce PostgreSQL-only `pg_trgm` / `GinIndex` while the project remains SQLite-only.

**Safe current mitigations:** strip/bound the search string, keep pagination, and document this limit.

**Deferred production work (when moving to PostgreSQL):**

1. Point `DATABASES` at PostgreSQL and add a driver (e.g. `psycopg`).
2. Enable the `pg_trgm` extension.
3. Add a trigram/`GinIndex` (or equivalent) migration for `movie_name` so substring search can scale.
4. Keep the same `icontains` API in Django so templates/URLs stay unchanged.

## Compromised key / git history

If a secret key was committed publicly, treat it as permanently compromised: rotate everywhere, then scrub history (for example `git filter-repo --replace-text`) and force-push only after coordinating with collaborators. GitHub may still retain the old blobs until support/cache expiry; rotation remains mandatory.
