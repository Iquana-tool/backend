# Flask Backend for Coral Analysis

## 📌 Overview
This repository contains a Flask backend for coral analysis, integrating **Segment Anything Model (SAM)** for image segmentation. It follows a modular structure with clear separation between **routes, services, models, and schemas** to ensure scalability and maintainability.

## 🏗️ Project Structure
```
app/
│── data             # Data folder. Saves images and CT Scans
│   |── meso-scale   # Coral images 
│   |── micro-scale  # Coral CT Scans
│── database         # Database models (SQLAlchemy). Define the database scheme
│   |── coral.py     
│── schemas          # API request/response validation (Marshmallow). Define the data format for requests/responses
│   │── *.py         
│── routes/          # API endpoints (Blueprints)
│   │── *.py         
│── services/        # Defines actual functionalities. Disconnected from routes.
│   │── *.py  
│── __init__.py      # Flask app factory function. Builds the app
│── config.py        # Configuration settings (database, environment variables)
migrations/          # Database migration scripts (Flask-Migrate) = Source control for db
.env                 # Environment variables (e.g., DB credentials)
requirements.txt     # Project dependencies
Readme.md            # Project documentation
run.py               # Entry point to start the Flask application
```

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone <repo-url>
cd <repo-folder>
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a **.env** file in the project root and define required environment variables:
```ini
FLASK_ENV=development
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///database.db
UPLOAD_FOLDER=uploads/
```

### 5️⃣ Initialize the Database
```sh
flask db init
flask db migrate -m "Initial migration."
flask db upgrade
```

### 6️⃣ Run the Flask Application
```sh
python run.py
```
Server will be accessible at: `http://127.0.0.1:5000/`

## 🛠️ API Endpoints
### 🔹 Segment an Image
TODO: Add this.

## 🏗️ Understanding the Structure
### **1. Data folder (`data/`)**
- Contains the data in two subfolders:
  - `meso-scale/` for images of corals
  - `micro-scale/` for CT scans (TODO: Add more details for file structure!)

### **2. Database schemas (`database/*.py`)**
- Defines **database tables** using SQLAlchemy.
- Example: `SegmentedImage` stores segmentation results.

### **3. API Schemas (`schemas/*.py`)**
- Handles **API request validation & response formatting** using Marshmallow.
- Ensures data consistency in API responses.

### **4. Routes (`routes/*.py`)**
- **Handles API requests** and calls services.
- Uses schemas for JSON serialization.

### **5. Services (`services/*.py`)**
- Implements **business logic** for backend functionality.
- **Does not handle API requests directly**.

## 📜 License
This project is licensed under the AGPL3 License.

---

