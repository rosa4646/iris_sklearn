import numpy as np
from IPython.display import display 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_data['data'], iris_data['target'], random_state=0
)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[1, 5.9, 1, 9.2]])
predict = knn.predict(X_new)

print(format(iris_data['target_names'][predict]))
display(iris_data)
