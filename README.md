# Deep Learning on Structured Data  

## Overview  
This project focuses on predicting the popularity of online news articles using structured data. The goal is to leverage deep learning techniques to analyze features from the [Online News Popularity Dataset](https://archive.ics.uci.edu/dataset/332/online+news+popularity) and predict the number of social interactions (shares) for news articles.  

The project explores the application of neural networks on structured data and compares it with traditional machine learning methods.  

---

## Dataset  

- **Source**: [UCI Machine Learning Repository - Online News Popularity Dataset](https://archive.ics.uci.edu/dataset/332/online+news+popularity)  
- **Description**: The dataset includes various features of online news articles published by Mashable. The data consists of 61 attributes, such as:  
  - Number of words, links, and images  
  - Day of publication  
  - Social engagement metrics  
- **Target Variable**: The number of shares each article received.  

---

## Features of the Project  

1. **Data Preprocessing**:  
   - Feature selection and engineering (e.g., scaling, normalization).  
   - Handling missing data, outliers, and categorical variables.  

2. **Machine Learning Models** (Baseline):  
   - Regression models, such as Random Forest and Gradient Boosting, were implemented for baseline performance comparison.  

3. **Deep Learning Model**:  
   - A fully connected neural network built with TensorFlow/Keras.  
   - Optimization techniques, such as dropout, batch normalization, and learning rate scheduling, were used for better performance.  

4. **Evaluation Metrics**:  
   - Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score for model performance evaluation.  

---

## Technologies Used  

- **Python**: The primary programming language.  
- **TensorFlow**: For building and training deep learning models.  
- **Scikit-learn**: For preprocessing and machine learning baselines.  
- **Matplotlib/Seaborn**: For data visualization and performance plotting.  

---

## Setup  

### Prerequisites  

1. Install Python (>= 3.7).  
2. Install required libraries using pip:  
   ```bash  
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn  
   ```  

---

### Installation  

1. Clone the Repository:  
   ```bash  
   git clone https://github.com/krishnavyas36/Deep_learning_structured.git  
   cd Deep_learning_structured  
   ```  

2. Download the Dataset:  
   - Visit the [UCI Online News Popularity Dataset page](https://archive.ics.uci.edu/dataset/332/online+news+popularity).  
   - Download and extract the dataset, and place it in the project directory (`data/` folder).  

3. Run the Preprocessing Script:  
   ```bash  
   python preprocess.py  
   ```  

4. Train the Deep Learning Model:  
   ```bash  
   python train_dl.py  
   ```  

5. (Optional) Train the Baseline Machine Learning Models:  
   ```bash  
   python train_ml.py  
   ```  

---

## Usage  

1. **Training**:  
   Modify the hyperparameters in `train_dl.py` or `train_ml.py` and run the respective scripts to train the models.  

2. **Evaluation**:  
   Evaluate the trained model on test data using the evaluation script:  
   ```bash  
   python evaluate.py  
   ```  

3. **Prediction**:  
   Use the prediction script to make predictions on new structured data:  
   ```bash  
   python predict.py --data_path <path_to_data_file>  
   ```  

---

## Results  

- **Baseline Machine Learning Models**:  
  - Random Forest achieved an R² score of X.XX and MSE of X.XX.  

- **Deep Learning Model**:  
  - The fully connected neural network outperformed traditional models with an R² score of X.XX and MSE of X.XX.  

*Note: Replace X.XX with actual results.*  

### Sample Outputs:  
Include performance metrics, loss/accuracy plots, and any key visualizations here.  

---

## Contributing  

If you’d like to contribute to this project, feel free to:  
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/YourFeature`).  
3. Commit your changes (`git commit -m 'Add YourFeature'`).  
4. Push to the branch (`git push origin feature/YourFeature`).  
5. Open a Pull Request.  

---

## License  

This project is for educational purposes. Please provide proper attribution if you use this work.  

---  

Let me know if you'd like to modify or add specific details!
