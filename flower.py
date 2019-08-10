import numpy as np
import pandas as pd

pd.set_option('display.width', 175)
pd.set_option('display.max_column', 8)
pd.set_option('precision', 2)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sbn
import warnings
warnings.filterwarnings(action='ignore')

hnames = ['sepal length' ,'sepal width' ,'petal length' ,'petal width','kind']
df = pd.read_csv('iris.csv',names = hnames)

print("\n\n",df.describe(include='all'))

print("\n Shape of the data set : ",df.shape)

print("\nCheck the count of null values for each feature :  \n", pd.isnull(df).sum())

print("\n\n", df['kind'].value_counts())

labels = ['Iris-setosa' , 'Iris-versicolor'  , 'Iris-virginica']
plt.pie(df['kind'].value_counts(),labels=labels , autopct='%.1f%%', explode=(0.05,0.05,0.05) , startangle=90 )

# Another way of plotting a pie chart :-
# df.kind.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(5,5))

arr = df.values
x = np.array(arr[:,0:4])
y = np.array(arr[:,4])


sbn.pairplot(df.dropna(),hue='kind')
plt.show()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y)
# print(y)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
x_scaled = sc.transform(x)


# Splitting the data

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_scaled , y ,test_size=0.3 , random_state=1 )
# print(y_train)


# Spot Check Algos

# 1) KNN

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor KNN Algo accuracy is : ",acc)

# 2) LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)*100
print("\nFor LDA Algo accuracy is : ",acc)

# 3) Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred , y_test)*100
print("\nFor Gaussian NB algo accuracy is: ",acc)

# 4) Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor Decision Tree Classifier Algo accuracy is: ",acc)

# 5) Random Forest

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor Random Forest Classifier Algo accuracy is: ",acc)

# 6) Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor SDG Classifier Algo accuracy is: ",acc)

# 7) Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor Gradient Boosting Classifier Algo accuracy is: ",acc)

# 8) SVM

from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor SVC Algo accuracy is: ",acc)

# 9) Linear SVC

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)*100
print("\nFor Linear SVC Algo accuracy is: ",acc)

# 10) Perceptron

from sklearn.linear_model import Perceptron
model = Perceptron()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)*100
print("\nFor Perceptron Algo accuracy is: ",acc)

# K-means Algo

from sklearn.cluster import KMeans

error=[]
for n in range(1,10):
    kmeans = KMeans(n_clusters=n , random_state= 0)
    kmeans.fit(x_train)
    error.append(kmeans.inertia_)

plt.plot(range(1,10) , error)
plt.scatter(3 , error[2] , c='r' , marker='x',label='Elbow')
plt.legend()
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

kmeans = KMeans(n_clusters=3 , random_state=1)
y_kmeans = kmeans.fit_predict(x_train)
print(y_kmeans)

plt.scatter(kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[: , 1] , s=10 , c='red' , label = 'Centroids')
plt.legend()
plt.show()
