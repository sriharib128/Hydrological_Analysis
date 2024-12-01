<!-- ### Repository Analysis

#### `src` Directory
The `src` directory contains the frontend application code:
- **Files**:
  - `App.css` and `index.css`: Stylesheets for the application.
  - `App.tsx` and `main.tsx`: Main TypeScript React components and entry points.
  - `vite-env.d.ts`: TypeScript declaration for Vite's environment variables.

- **Subdirectories**:
  - `components`: Likely houses reusable UI components.
  - `hooks`: Contains custom React hooks.
  - `lib`: Contains library modules or helper functions.

#### `server` Directory
The `server` directory contains backend-related files and additional resources:
- **Files**:
  - `app.py`: Likely the main Python script for running the server or application backend.
  - `clipped_ndvi_crs.tif`: A geospatial file, possibly related to hydrological data analysis.
  - `discharge_analysis.py`: Script for discharge or flow analysis.
  - `dqt_module.py`: A module, likely for data quality or transformation.
  - `ml_module.py`: Contains machine learning-related functionality.
  - `requirements.txt`: Specifies Python dependencies.

- **Subdirectory**:
  - `__pycache__`: Compiled Python files for faster execution.

---

### Suggested README

markdown -->
# Hydrological Analysis

This repository provides tools and components for hydrological analysis using a combination of web technologies and Python-based scientific computation. The project aims to integrate frontend interfaces with backend processing to analyze and visualize hydrological data.

## Features
- *Frontend*: Built with TypeScript and React, styled using Tailwind CSS, and powered by Vite for fast builds.
- *Backend*: Python-based backend for hydrological computations, including discharge analysis, machine learning models, and geospatial data processing.
- *Geospatial Data*: Support for handling .tif files for remote sensing and hydrological datasets.
- *Extensibility*: Modular design in both frontend and backend for easy customization.

## Project Structure

### Frontend (src)
- *Components*: Reusable React components.
- *Hooks*: Custom hooks for managing state and side effects.
- *Styling*: Tailwind CSS for rapid UI development.
- *Build Tool*: Vite for development and production builds.

### Backend (server)
- *Scripts*:
  - app.py: Main server script to handle API requests.
  - discharge_analysis.py: Tools for analyzing water discharge data.
  - ml_module.py: Machine learning utilities for hydrological predictions.
  - dqt_module.py: Data quality and transformation utilities.
- *Data*: Includes geospatial data (clipped_ndvi_crs.tif) for analysis.
- *Dependencies*: Managed via requirements.txt.

## Installation

### Prerequisites
- Node.js and npm/yarn for the frontend.

### Setup

#### 1. Clone the Repository
```
git clone https://github.com/sriharib128/Hydrological_Analysis.git
cd hydrological-analysis
```


#### 2. Install Frontend Dependencies
```
cd src
npm install
```

#### 3. Install Backend Dependencies
```
cd ../server
pip install -r requirements.txt
```

## Usage

### Running the Frontend
```
cd src
npm run dev
```

Access the application at http://localhost:3000.

### Running the Backend
```
cd server
python app.py
```
The backend will run on http://localhost:5000.
