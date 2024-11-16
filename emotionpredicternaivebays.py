import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\adith\OneDrive\Desktop\Projects\ML\archive\Emotion_classify_Data.csv")

x = dataset['Comment']
y = dataset['Emotion']

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Naive Bayes Classifier')
plt.show()

new_comment=input("Enter something")
new_comment_transformed = vectorizer.transform([new_comment])
predicted_emotion = model.predict(new_comment_transformed)
print(f"Predicted Emotion: {predicted_emotion}")