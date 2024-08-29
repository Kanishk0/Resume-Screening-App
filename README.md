Here’s a README file for your project:

---

# Resume Screening with Python

## Overview
This project is designed to automate the resume screening process using Python. The main objective is to classify resumes into predefined job categories based on their content, leveraging natural language processing (NLP) and machine learning techniques.

## Features
- **Resume Parsing**: Processes and cleans resumes to extract relevant information.
- **Text Processing**: Implements techniques like TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction from resumes.
- **Model Training**: Trains a machine learning classifier to categorize resumes into various job roles.
- **Prediction**: Predicts the job category of a given resume using the trained model.

## Dependencies
- Python 3.x
- numpy
- pandas
- scikit-learn
- nltk
- pickle

## Files
- `Resume Screening with Python.ipynb`: The main Jupyter notebook containing all the code for resume screening.
- `clf.pkl`: Pre-trained machine learning classifier.
- `tfidfd.pkl`: TF-IDF vectorizer used for feature extraction.
- `sample_resume.txt`: A sample resume file for testing.

## How to Run
1. **Install Dependencies**: Ensure you have all the required Python libraries installed. You can install them using pip:
   ```bash
   pip install numpy pandas scikit-learn nltk
   ```
2. **Open the Notebook**: Launch Jupyter Notebook and open the `Resume Screening with Python.ipynb` file.
3. **Run the Notebook**: Execute the cells in the notebook to load the model, process the resume, and predict its job category.
4. **Test with a Sample Resume**: Replace `myresume` with the content of any resume you wish to categorize.

## Example Usage
```python
# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))

# Clean the input resume
cleaned_resume = cleanResume(myresume)

# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidfd.transform([cleaned_resume])

# Make the prediction using the loaded classifier
prediction_id = clf.predict(input_features)[0]

# Get the predicted category
category_name = category_mapping.get(prediction_id, "Unknown")
print("Predicted Category:", category_name)
```

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests.
