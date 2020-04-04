from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import adjusted_rand_score as ars
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
model = KMeans(n_clusters = 10)
model.fit(X_train)
y_pred = model.fit_predict(X_test)
print(ars(y_test,y_pred))

