import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

training_file_path = "datasets/training_data.csv"
df_train = pd.read_csv(training_file_path)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(df_train["Pergunta"])
y_train = df_train["Categoria"]

model = MultinomialNB()
model.fit(X_train, y_train)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Modelo treinado e salvo com sucesso!")
