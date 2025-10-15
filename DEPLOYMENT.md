# Application Screenshots & Deployment Guide

## Streamlit Application Preview

### Main Interface

The Heart Disease Predictor features a professional, user-friendly interface with three main sections:

#### 1. **Prediction Tab** 🔍
- **Model Selection**: Choose from Logistic Regression, Random Forest, XGBoost, or Best Model
- **Input Form**: Enter patient clinical data across three columns for easy data entry
- **Prediction Button**: Large, prominent button to trigger the prediction
- **Results Display**: Clear visualization of prediction results with:
  - Risk level indicator (Low Risk / High Risk)
  - Probability breakdown with metrics
  - Bar chart showing probability distribution
  - Medical disclaimer

#### 2. **Model Performance Tab** 📈
- Overview of model evaluation metrics
- Links to detailed analysis in Jupyter notebook
- Description of metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

#### 3. **About Dataset Tab** ℹ️
- Comprehensive feature descriptions
- Dataset statistics
- Target variable information

### Application Features

✅ **Professional UI Design**
- Custom CSS styling with medical-themed colors
- Responsive layout that works on all screen sizes
- Clear visual hierarchy

✅ **Interactive Elements**
- Dropdown selections for categorical variables
- Number inputs with validation
- Real-time model switching
- Color-coded risk indicators

✅ **User Experience**
- Intuitive three-column layout for data entry
- Helpful tooltips and descriptions
- Medical disclaimers and safety information
- Clear result visualization

## Deployment Options

### 1. Streamlit Cloud (Recommended)

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Set main file: `app/streamlit_app.py`
7. Click "Deploy"

**Advantages:**
- Free hosting
- Automatic updates on git push
- Easy sharing via URL
- No server management

### 2. Heroku

**Steps:**
1. Create a `Procfile`:
   ```
   web: streamlit run app/streamlit_app.py --server.port=$PORT
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### 3. AWS EC2

**Steps:**
1. Launch an EC2 instance
2. Install Python and dependencies
3. Clone your repository
4. Run: `streamlit run app/streamlit_app.py --server.port 80`
5. Configure security groups for port 80

### 4. Google Cloud Platform

**Steps:**
1. Create a new project
2. Enable App Engine
3. Create `app.yaml`:
   ```yaml
   runtime: python39
   entrypoint: streamlit run app/streamlit_app.py --server.port 8080
   ```
4. Deploy: `gcloud app deploy`

### 5. Docker

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py"]
```

**Build and run:**
```bash
docker build -t heart-disease-predictor .
docker run -p 8501:8501 heart-disease-predictor
```

## Environment Variables

If needed for deployment, create `.streamlit/config.toml`:

```toml
[server]
headless = true
enableCORS = false
port = 8501

[theme]
primaryColor = "#E74C3C"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Production Considerations

### Security
- Always display medical disclaimers
- Don't store sensitive patient data
- Use HTTPS in production
- Implement authentication if needed

### Performance
- Models are cached using `@st.cache_resource`
- Consider model compression for faster loading
- Use CDN for static assets

### Monitoring
- Enable Streamlit analytics
- Monitor application errors
- Track prediction accuracy over time

## Support

For deployment issues:
1. Check [Streamlit documentation](https://docs.streamlit.io/)
2. Review platform-specific guides
3. Open an issue on GitHub
