# fake-news-detection
This project is a Machine Learning-based Fake News Detection system that classifies news articles as REAL or FAKE using Natural Language Processing techniques. The dataset is preprocessed by cleaning text and converting it into numerical features using TF-IDF vectorization. A Logistic Regression model is trained on the processed data to learn patterns that distinguish fake news from real news. The model is evaluated using accuracy and classification reports.

The project also includes a Streamlit web application where users can enter news text and get instant predictions along with confidence scores. It demonstrates an end-to-end ML pipeline including data preprocessing, model training, evaluation, saving the model, and deployment.

Technologies used include Python, Pandas, NumPy, Scikit-learn, NLTK, and Streamlit. To run the project, install dependencies using pip install -r requirements.txt, train the model using python train.py, and start the app using streamlit run app.py.
