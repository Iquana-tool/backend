# FastAPI Backend for Coral Analysis

## 📌 Overview
This repository contains a FastAPI backend for coral analysis, integrating **Segment Anything Model (SAM)** for image segmentation. It follows a modular structure with clear separation between **routes, services, models, and schemas** to ensure scalability and maintainability.

## 🏗️ Project Structure
```
app/
│── database         # Database models (SQLAlchemy). Define the database scheme
│   |── coral.py     
│── schemas          # API request/response validation (PyDantic). Define the data format for requests/responses
│   │── *.py         
│── routes/          # API endpoints (FastAPI Routers)
│   │── *.py         
│── services/        # Defines actual functionalities. Disconnected from routes.
│   │── *.py  
│── __init__.py      # FastAPI app factory function. Builds the app and initialises the database
data/                # Data folder. Saves images and CT Scans, and all other kind of data
|── meso-scale       # Coral images and embeddings
|── micro-scale      # Coral CT Scans 
|── database.db      # SQLite database
tests/*              # Tests for services and routes
config.py            # Configuration settings (database, environment variables, paths etc.)
requirements.txt     # Project dependencies
Readme.md            # This file
main.py              # Entry point to start the Fastapi application
```

## 🛠️ Installation & Setup
### Option 1: Using [podman](https://podman.io)
Install podman and build the image using
```sh
podman build -t coral-backend .
```
Then run the container using
```sh
podman run -p 8000:8000 coral-backend
```
### Option 2: Using venv
#### 1. Clone the Repository
```sh
git clone <repo-url>
cd <repo-folder>
```

#### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

#### 3. Install Dependencies
> **Note:** If you want to use CUDA for GPU acceleration, make sure to install the correct version of PyTorch that 
> matches your CUDA version **before** installing the requirements. You can find the correct command for your system 
> [here](https://pytorch.org/get-started/locally/). The requirements.txt file contains the CPU version of PyTorch, which
> will be installed by default, if you do not install the CUDA version first.

Install the required packages:
```sh
pip install -r requirements.txt
```


#### 4. Run the Flask Application
```shell
fastapi main.py
```
or for developer mode
```shell
fastapi dev main.py
```
>Server will be accessible at: http://127.0.0.1:8000
>
>Swagger Docs can be accessed at: http://127.0.0.1:8000/docs
---
## 🏗️ Understanding the Structure
### **1. Data folder (`data/`)**
- Contains the data in two subfolders:
  - `meso-scale/` for images of corals
  - `micro-scale/` for CT scans (TODO: Add more details for file structure!)

### **2. Database schemas (`database/*.py`)**
- Defines **database tables** using SQLAlchemy.
- Example: `SegmentedImage` stores segmentation results.

### **3. API Schemas (`schemas/*.py`)**
- Handles **API request validation & response formatting** using PyDantic.
- Ensures data consistency in API responses.

### **4. Routes (`routes/*.py`)**
- **Handles API requests** and calls services.
- **Does not implement functional code**.
- Uses schemas for JSON serialization.
- Example: `sam2_endpoints.py` contains routes for image segmentation.

### **5. Services (`services/*.py`)**
- Implements **business logic** for backend functionality. 
- **Does not handle API requests directly**.
- Disconnected from routes for better maintainability.
- Example: `sam2.py` contains the SAM model for image segmentation.

## Understanding the workflow
1. **API Request**: User sends a request to the API endpoint.
2. **Route**: Receives the request, validates it using *schemas*, and calls the *service*.
   1. **Schema**: Validates the request data and extracts into a usable format.
   2. **Service**: Implements the business logic, interacts with the database, does computations and returns the result.
   3. **Database**: Stores and retrieves data using SQLAlchemy models.
   4. **Service**: Returns the result to the route.
   5. **Schema**: Puts the result in valid JSON response format.
3. **Route**: Sends the response back to the user.
4. Done!
---
## Endpoint Scheme
# Warning: This is not updated!
### **Authentication**
> Do we need this?

### **Coral**

> **GET** `/coral/get_coral`
> - Parameters:
>   - `coral_id` (int): Coral ID
> - Get all coral parameters by ID.
> - Response: Coral parameters.

> **POST** `/coral/add_coral`
> - Parameters:
>   - `metadata` (dict): Coral metadata
> - Add a new coral to the database.
> - Response: Coral ID.
### **Images**
> **GET** `/images/get_image` 
>- Parameters: 
>  - `image_id` (int): Image ID
>- Get an image by ID. Used for displaying images.
>- Response: Image file.

> **POST** '/images/upload_meso_image' 
>- Parameters: 
>  - `file` (file): Image file
>  - `coral_id` (int): Coral id to which the image belongs (? this might be incorrect)
>- Upload an image. Used for uploading meso-scale images.
>- Response: Image ID. 

> **POST** '/images/upload_ct_scan'
> - Parameters:
>   - `folder` (folder): Folder of CT scans
>   - `coral_id` (int): Coral id to which the CT scan belongs
> - Upload a folder of CT scans. Used for uploading micro-scale images.
> - Response: Scan ID.
> - Note: This endpoint will be used for uploading a folder of CT scans. The folder should contain the CT scans and a 
> log file containing the scan parameters.

### **Segmentation**
> **POST** `/segmentation/auto_segment_image`
> - Parameters:
>   - `image_id` (int): Image ID
>   - `model` (str): Model to use for segmentation
> - Automatically segment an image using the specified model.
> - Response: Segmented image. Returns cached result if available.

> **POST** `/segmentation/prompt_segment_image`
> - Parameters:
>   - `image_id` (int): Image ID
>   - `model` (str): Model to use for segmentation
>   - `point_prompts` (List[Tuple[float, float, int]]): Point prompts. A list of (x, y, label) tuples, where the label 
>   is one of background (0) or foreground (1).
>   - `box_prompts` (List[Tuple[float, float, float, float]]): Box prompts. A list of (x1, y1, x2, y2). Everything 
>   inside the box is considered foreground.
>   - `polygon_prompts` (List[List[Tuple[float, float]]]): Polygon prompts. A list of lists of (x, y) tuples. Everything
>   inside the polygon is considered foreground.
> - Segments the image given the prompts. Used for interactive segmentation.
> - Note: All coordinates should be in the range [0, 1]. Each type of prompt is optional. However, at least one prompt
> should be provided.
> - Response: Segmented image.

> **POST** `/segmentation/auto_segmented_scan`
> TBD
## 📜 License
This project is licensed under the AGPL3 License.

---

