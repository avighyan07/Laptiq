from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
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
from werkzeug.security import generate_password_hash, check_password_hash

nltk.download('punkt')

# ======================
# App initialization
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
# Global constants
# ======================
USER_FILE = 'users.csv'
DATA_FILE = os.path.join(os.path.dirname(__file__), 'reviews.csv')
expected_columns = ['company', 'model_name', 'user_name', 'rating', 'review_text', 'sentiment', 'date', 'aspect_sentiments']

# Initialize users.csv if not exists
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=['username', 'email', 'password']).to_csv(USER_FILE, index=False)

# Load or initialize reviews data
try:
    df = pd.read_csv(DATA_FILE)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    df = df[expected_columns]
    print(f"✅ Loaded {len(df)} reviews from {DATA_FILE}")
except Exception as e:
    print("❌ Loading reviews.csv failed:", e)
    df = pd.DataFrame(columns=expected_columns)

# ======================
# Load ML models
# ======================
vectorizer = joblib.load('models/vectorizer.joblib')
sentiment_model = joblib.load('models/sentiment_model.joblib')
pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df_laptop = pickle.load(open('models/df.pkl', 'rb'))

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
        if len(slug_parts) >= 2:
            return slug_parts[0], slug_parts[1]
        elif slug_parts:
            return slug_parts[0], ''
        return None, None
    except Exception as e:
        print("URL extraction error:", e)
        return None, None

aspects = ['battery', 'display', 'performance', 'design', 'keyboard', 'price', 'camera', 'sound']

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
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    wc.to_file(f'static/{filename}')
    plt.close()

# ======================
# Routes
# ======================
@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', username=username)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        existing_username = User.query.filter_by(username=username).first()
        if existing_username:
            flash('Username already exists. Please choose another.', 'danger')
            return render_template('signup.html')

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already exists. Please login.', 'danger')
            return render_template('signup.html')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        pd.DataFrame([{
            'username': username,
            'email': email,
            'password': hashed_password
        }]).to_csv(USER_FILE, mode='a', header=not os.path.exists(USER_FILE), index=False)

        login_user(new_user)
        session['username'] = username
        flash('Signup successful!', 'success')
        return redirect(url_for('index'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['username'] = user.username
            flash('Login successful.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/submit_review', methods=['GET', 'POST'])
def submit_review():
    global df
    message = ''
    if request.method == 'POST':
        url = request.form.get('url')
        user_name = request.form.get('user_name')
        rating_str = request.form.get('rating')
        review_text = request.form.get('review_text')

        rating = int(rating_str) if rating_str else None
        company, model_name = (extract_model_from_url(url) if url and url.strip()
                               else (request.form.get('company'), request.form.get('model_name')))

        if company and model_name and user_name and review_text and rating:
            X = vectorizer.transform([review_text])
            sentiment = sentiment_model.predict(X)[0]
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            aspect_sentiments = extract_aspect_sentiments(review_text)
            aspect_sentiments_str = str(aspect_sentiments)

            new_review = pd.DataFrame([[company, model_name, user_name, rating, review_text, sentiment, date, aspect_sentiments_str]],
                                      columns=expected_columns)
            df = pd.concat([df, new_review], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)

            message = 'Review submitted successfully with aspect sentiments!'
        else:
            message = 'Please fill all fields correctly.'

    return render_template('submit_review.html', message=message)

@app.route('/dashboard')
def dashboard():
    df_latest = pd.read_csv(DATA_FILE)

    pos_reviews = df_latest[df_latest['sentiment'] == 2]['review_text'].str.cat(sep=' ')
    neg_reviews = df_latest[df_latest['sentiment'] == 1]['review_text'].str.cat(sep=' ')

    if pos_reviews.strip():
        generate_wordcloud(pos_reviews, 'pos_wordcloud.png', colormap='Greens')

    if neg_reviews.strip():
        generate_wordcloud(neg_reviews, 'neg_wordcloud.png', colormap='Reds')

    grouped = df_latest.groupby(['company', 'model_name'])
    products = []
    for (company, model_name), group in grouped:
        avg_rating = group['rating'].mean()
        sentiment_counts = group['sentiment'].value_counts().to_dict()
        recent_reviews = group.tail(3).to_dict(orient='records')
        rating_counts = group['rating'].value_counts().sort_index().to_dict()
        for i in range(1, 6):
            if i not in rating_counts:
                rating_counts[i] = 0
        products.append({
            'company': company,
            'model_name': model_name,
            'avg_rating': avg_rating,
            'sentiment_counts': sentiment_counts,
            'recent_reviews': recent_reviews,
            'rating_counts': rating_counts
        })

    return render_template('dashboard.html', products=products)

@app.route('/predict_form', methods=['GET'])
def predict_form():
    companies = ['Dell', 'HP', 'Apple']
    typenames = ['Notebook', 'Gaming']
    cpu_brands = ['Intel Core i5', 'Intel Core i7']
    gpu_brands = ['Nvidia', 'AMD']
    oss = ['Windows', 'Mac']
    return render_template('predict_form.html',
                           companies=companies,
                           typenames=typenames,
                           cpu_brands=cpu_brands,
                           gpu_brands=gpu_brands,
                           oss=oss)

@app.route('/predict_price', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':
        company = request.form['company']
        typename = request.form['typename']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = int(request.form['touchscreen'])
        ips = int(request.form['ips'])
        ppi = float(request.form['ppi'])
        cpu_brand = request.form['cpu_brand']
        gpu_brand = request.form['gpu_brand']
        os = request.form['os']

        input_df = pd.DataFrame([{
            'Company': company,
            'TypeName': typename,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': touchscreen,
            'Ips': ips,
            'ppi': ppi,
            'Cpu brand': cpu_brand,
            'Gpu brand': gpu_brand,
            'bavii': gpu_brand,
            'os': os
        }])

        predicted_price = int(pipe.predict(input_df)[0])
        return render_template('predict.html', price=predicted_price)
    else:
        return redirect(url_for('predict_form'))

# ======================
# Run the app
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
