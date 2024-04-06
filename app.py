from flask import Flask, render_template, request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset and perform preprocessing steps
df = pd.read_csv('F:\ML PROJECTS\JobPlacementDataset - Sem6 Project\Job_Placement_Data.csv', usecols=[0,1,3,4,5,6,7,8,9,10,11,12])  # Update with your actual dataset file path

# Impute missing values for numerical columns with mean
numeric_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'emp_test_percentage', 'mba_percent']
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute missing values for categorical columns with most frequent category (mode)
categorical_cols = ['gender', 'hsc_board', 'hsc_subject', 'undergrad_degree', 'work_experience', 'specialisation']
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

x_train = df.drop('status', axis=1)
y_train = df['status']

# Define preprocessing pipeline
all_cols = set(x_train.columns)
non_categorical_cols = list(all_cols.difference(categorical_cols))

preprocessor = ColumnTransformer(
    transformers=[
        ('tnf', OneHotEncoder(sparse=False, drop='first'), categorical_cols),
        ('scaler', StandardScaler(), non_categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the pipeline on training data
pipeline.fit(x_train, y_train)

# Flask route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle form submission and display result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        # Assuming you have received the form data from the user
        gender = request.form['gender']
        ssc_percentage = float(request.form['ssc_percentage'])
        hsc_percentage = float(request.form['hsc_percentage'])
        hsc_board = request.form['hsc_board']
        hsc_subject = request.form['hsc_subject']
        degree_percentage = float(request.form['degree_percentage'])
        undergrad_degree = request.form['undergrad_degree']
        work_experience = request.form['work_experience']
        emp_test_percentage = float(request.form['emp_test_percentage'])
        specialisation = request.form['specialisation']
        mba_percent = float(request.form['mba_percent'])

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'gender': [gender],
            'ssc_percentage': [ssc_percentage],
            'hsc_percentage': [hsc_percentage],
            'hsc_board': [hsc_board],
            'hsc_subject': [hsc_subject],
            'degree_percentage': [degree_percentage],
            'undergrad_degree': [undergrad_degree],
            'work_experience': [work_experience],
            'emp_test_percentage': [emp_test_percentage],
            'specialisation': [specialisation],
            'mba_percent': [mba_percent],
        })

        # Use the trained model to predict probability
        probability = pipeline.predict_proba(user_input)[:, 1]  # Probability of getting placed

        # Render the result page with the probability
        return render_template('result.html', probability=round(probability[0] * 100, 2))


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
