# ======================
# Import dependencies
# ======================
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import pandas as pd
import joblib
import os
from datetime import datetime
from urllib.parse import urlparse
import pickle
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# ======================
# App setup
# ======================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'any-secure-random-key'

db = SQLAlchemy(app)

# ======================
# Flask-Login setup
# ======================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ======================
# Download NLTK punkt
# ======================
nltk.download('punkt')

# ======================
# User model
# ======================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    is_admin = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ======================
# Aspects for ABSA
# ======================
aspects = ['battery', 'display', 'performance', 'design', 'keyboard', 'price', 'camera', 'sound']

# ======================
# Load models
# ======================
vectorizer = joblib.load('models/vectorizer.joblib')
sentiment_model = joblib.load('models/sentiment_model.joblib')
pipe = pickle.load(open('models/pipe.pkl', 'rb'))

# ======================
# Data file setup
# ======================
DATA_FILE = '/tmp/reviews.csv'  # use /tmp for Render deployment stability

expected_columns = ['company', 'model_name', 'user_name', 'rating', 'review_text', 'sentiment', 'date', 'aspect_sentiments']

# Initialize reviews.csv if not exists
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=expected_columns).to_csv(DATA_FILE, index=False)

# ======================
# Helper functions
# ======================
def extract_model_from_url(url):
    try:
        path = urlparse(url).path
        parts = path.split('/')
        for part in parts:
            if part and not part.startswith('dp'):
                slug = part
                break
        else:
            return None, None
        slug_parts = slug.split('-')
        return (slug_parts[0], slug_parts[1]) if len(slug_parts) >= 2 else (slug_parts[0], '')
    except Exception as e:
        print("URL extraction error:", e)
        return None, None

def extract_aspect_sentiments(review):
    aspect_sentiments = {}
    tokens = nltk.word_tokenize(review.lower())
    blob = TextBlob(review)
    for aspect in aspects:
        if aspect in tokens:
            for sentence in blob.sentences:
                if aspect in sentence.lower():
                    polarity = sentence.sentiment.polarity
                    aspect_sentiments[aspect] = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
    return aspect_sentiments

def generate_wordcloud(text, filename, colormap='autumn'):
    wc = WordCloud(width=800, height=400, background_color='black', colormap=colormap).generate(text)
    wc.to_file(f'static/{filename}')
    plt.close()

# ======================
# Routes
# ======================

@app.route('/')
def index():
    return render_template('index.html')

# -------- Signup --------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if User.query.filter_by(username=username).first():
            flash('Username exists.', 'danger')
            return render_template('signup.html')
        if User.query.filter_by(email=email).first():
            flash('Email exists.', 'danger')
            return render_template('signup.html')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        flash('Signup successful.', 'success')
        return redirect(url_for('index'))
    return render_template('signup.html')

# -------- Login --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

# -------- Logout --------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'success')
    return redirect(url_for('index'))

# -------- Submit review --------
@app.route('/submit_review', methods=['GET', 'POST'])
def submit_review():
    message = ''
    if request.method == 'POST':
        url = request.form.get('url')
        user_name = request.form.get('user_name')
        rating_str = request.form.get('rating')
        review_text = request.form.get('review_text')
        rating = int(rating_str) if rating_str else None
        company, model_name = extract_model_from_url(url) if url and url.strip() else (None, None)

        df = pd.read_csv(DATA_FILE)

        if company and model_name and user_name and review_text and rating:
            X = vectorizer.transform([review_text])
            sentiment = sentiment_model.predict(X)[0]
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            aspect_sentiments = extract_aspect_sentiments(review_text)
            new_review = pd.DataFrame([{
                'company': company, 'model_name': model_name, 'user_name': user_name,
                'rating': rating, 'review_text': review_text, 'sentiment': sentiment,
                'date': date, 'aspect_sentiments': str(aspect_sentiments)
            }])
            df = pd.concat([df, new_review], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            message = 'Review submitted successfully.'
        else:
            message = 'Please fill all fields correctly.'
    return render_template('submit_review.html', message=message)

# -------- Dashboard --------
@app.route('/dashboard')
def dashboard():
    df = pd.read_csv(DATA_FILE)
    pos_reviews = df[df['sentiment'] == 2]['review_text'].str.cat(sep=' ')
    neg_reviews = df[df['sentiment'] == 1]['review_text'].str.cat(sep=' ')
    if pos_reviews.strip(): generate_wordcloud(pos_reviews, 'pos_wordcloud.png', 'Greens')
    if neg_reviews.strip(): generate_wordcloud(neg_reviews, 'neg_wordcloud.png', 'Reds')

    products = []
    for (company, model_name), group in df.groupby(['company', 'model_name']):
        products.append({
            'company': company,
            'model_name': model_name,
            'avg_rating': group['rating'].mean(),
            'sentiment_counts': group['sentiment'].value_counts().to_dict(),
            'recent_reviews': group.tail(3).to_dict(orient='records'),
            'rating_counts': {i: group['rating'].value_counts().get(i, 0) for i in range(1, 6)}
        })
    return render_template('dashboard.html', products=products)

# -------- Search reviews --------
@app.route('/search_reviews', methods=['GET', 'POST'])
def search_reviews():
    results = []
    query = request.form.get('query', '').lower() if request.method == 'POST' else ''
    if query:
        df = pd.read_csv(DATA_FILE)
        mask = df['user_name'].str.lower().str.contains(query) | df['review_text'].str.lower().str.contains(query) | df['date'].str.contains(query)
        results = df[mask].to_dict(orient='records')
    return render_template('search_reviews.html', results=results, query=query)

# ======================
# Run app
# ======================
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
