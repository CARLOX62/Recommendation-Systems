<h1 align="center">🚀 Recommendation System</h1>
<h3 align="center">A Multi-Domain Intelligent Recommendation Web App</h3>

<p align="center">
  <img src="https://img.shields.io/github/stars/CARLOX62/Recommendation-System?style=for-the-badge" />
  <img src="https://img.shields.io/github/forks/CARLOX62/Recommendation-System?color=blue&style=for-the-badge" />
  <img src="https://img.shields.io/github/issues/CARLOX62/Recommendation-System?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/CARLOX62/Recommendation-System?style=for-the-badge" />
</p>

---

## 🎯 Overview

This project is a fully functional Recommendation System built using **Flask, Machine Learning, and APIs**.  
It provides personalized suggestions for:

- 📚 **Books**
- 🎬 **Movies**
- 🎧 **Music**
- 👗 **Fashion Items**

---

## ✨ Features

- 🔍 Personalized smart recommendations  
- 🎞️ Movie details with posters & ratings via **TMDB**
- 🎶 Spotify-powered content similarity
- 🎨 Deep learning-based Fashion similarity search
- 🧠 Machine learning + Deep learning hybrid models
- ☁ Google Drive auto-download support for large files

---

## 🧠 Tech Stack

| Category | Technology |
|---------|------------|
| Backend Framework | Flask |
| Machine Learning | Scikit-Learn, TensorFlow |
| Feature Extraction | ResNet50 |
| APIs | Spotify API, TMDB API |
| Storage | Pickle Models, Google Drive |
| Frontend | HTML, CSS, JavaScript, Bootstrap |

---

## 📁 Project Structure

```
📦 Recommendation-System
├── Book Recommendation
│   ├── books.pkl
│   ├── similarity.pkl
│   └── app.py
├── Movie Recommendation
│   ├── movies.pkl
│   ├── similarity.pkl
│   └── app.py
├── Music Recommendation
│   ├── spotify_model.pkl
│   ├── data.csv
│   └── app.py
├── Fashion Recommendation
│   ├── images/
│   ├── embeddings.pkl
│   └── app.py
├── static/  (CSS/JS/Images)
├── templates/ (HTML Files)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset & Model Access (Google Drive Support)

Since model files are large, they are stored in **Google Drive** and downloaded automatically when needed.

### 🔽 Automatic Download Script

```python
import gdown, os, pickle

os.makedirs("models", exist_ok=True)

files = {
    "books.pkl": "https://drive.google.com/uc?id=FILE_ID_1",
    "similarity.pkl": "https://drive.google.com/uc?id=FILE_ID_2",
    "movies.pkl": "https://drive.google.com/uc?id=FILE_ID_3",
    "spotify_model.pkl": "https://drive.google.com/uc?id=FILE_ID_4",
    "fashion_embeddings.pkl": "https://drive.google.com/uc?id=FILE_ID_5"
}

for filename, url in files.items():
    save_path = f"models/{filename}"
    if not os.path.exists(save_path):
        print(f"📥 Downloading {filename}...")
        gdown.download(url, save_path, quiet=False)

print("✅ All required model files are ready!")
```

| File Name | Type | Purpose |
|----------|------|---------|
| books.pkl | Pickle | Book model |
| similarity.pkl | Pickle | Book similarity matrix |
| movies.pkl | Pickle | Movie features |
| spotify_model.pkl | Pickle | Music recommendation model |
| fashion_embeddings.pkl | Pickle | Deep learning embeddings |

---

## 🛠️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/CARLOX62/Recommendation-System.git
cd Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

---

## 🎥 Live UI Preview

| Home Page | Recommendation Output |
|-----------|-----------------------|
| <img width="1920" height="1080" alt="Screenshot (367)" src="https://github.com/user-attachments/assets/dfd1907c-4c26-445e-b50f-90430b1f4377" /> | <img width="1920" height="1080" alt="Screenshot (368)" src="https://github.com/user-attachments/assets/ca38d0c8-7311-4696-af8e-73cb3421e95e" />
 <img width="1920" height="1080" alt="Screenshot (369)" src="https://github.com/user-attachments/assets/691f2b1c-8246-45f3-a29b-2d5bd7e9097c" />
 <img width="1920" height="1080" alt="Screenshot (370)" src="https://github.com/user-attachments/assets/05f2f7f9-4121-4bb4-bb1b-d5bc52867025" />
 <img width="1920" height="1080" alt="Screenshot (371)" src="https://github.com/user-attachments/assets/c70cd101-c1e5-4894-8b19-5a40a03bdca9" /> 




---

## 🚀 Future Enhancements

- User login system  
- Personalized user history tracking  
- Cloud-based model hosting  
- Voice-based recommendation search  

---

## 👨‍💻 Author

**👋 Aniket Kumar**  
Machine Learning Engineer | Python Developer | AI Enthusiast  

📌 India 🇮🇳  
📫 Email: aniketkumarsonu62@gmail.com

### 🌐 Connect With Me

<p align="left">
<a href="https://github.com/CARLOX62"><img src="https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github" /></a>
<a href="https://www.linkedin.com/in/aniket-kumar-7b4104298/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin" /></a>
<a href="mailto:aniketkumarsonu62@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail" /></a>
</p>

---

## ⭐ Support

If you find this project useful, please consider **starring ⭐ the repo** — it helps others find it!

---

### 📝 License

This project is licensed under the **MIT License**.
