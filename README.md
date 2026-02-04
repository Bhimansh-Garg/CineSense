# CineSense 🎬

CineSense is a **Django-based web application** for analyzing and managing movie reviews.  
It includes a pre-trained sentiment analysis model (stored in `review/models/`) and a simple UI for interacting with reviews.

---

## ✨ Features
- Submit and analyze movie reviews.
- Built with **Django 5.x**.
- Uses pre-trained ML models (included in repo).
- Includes Jupyter Notebook (`main.ipynb`) for experimentation and model training.
- SQLite database (`db.sqlite3`) for local storage.

---

## 🛠 Tech Stack
- **Backend**: Django 5.x (Python 3.12+)
- **Frontend**: Django Templates, HTML, CSS
- **Database**: SQLite (default)
- **Other**: Pillow (for image handling)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Bhimansh-Garg/CineSense.git
cd CineSense
```
## Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```
## Install Dependencies
```bash
pip install -r requirements.txt
```
## Apply Migrations
```bash
python manage.py migrate
```

## Run the Server
```bash
python manage.py runserver
```
Open 👉 http://127.0.0.1:8000/
 in your browser.

 
## 📂 Project Structure
```bash
cinesense/             # Main Django project folder
│   manage.py          # Django CLI
│   db.sqlite3         # SQLite database (auto-created)
│   requirement.txt    # Project dependencies
│
├── cinesense/         # Django settings & URLs
├── review/            # Review app (with ML model integration)
│   ├── models/        # Pre-trained model + tokenizer
│   ├── templates/     # HTML templates
│   └── ...
├── media/             # Uploaded images
├── imdb-dataset.../   # Dataset (optional, only for retraining)
├── main.ipynb         # Notebook for ML training experiments
└── README.md          # Documentation
```


