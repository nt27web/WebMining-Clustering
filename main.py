import json
import requests

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sklearn.metrics.pairwise as pw
from IPython.display import display
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _recommend():


    data = pd.read_csv('HR-Employee-Attrition.csv')

    display(data.shape)
    # display(data.head(10))
    # display(data.isnull().sum())

    # display(data['Age'].describe())
    # display(data['DailyRate'].describe())
    # display(data['EducationField'].unique())
    #
    # display(data['YearsAtCompany'].describe())
    # display(data['YearsInCurrentRole'].describe())
    # display(data['YearsSinceLastPromotion'].describe())
    # display(data['YearsWithCurrManager'].describe())

    # display(len(f_data['Category'].unique()))
    # display(len(data))

    # return 0
    # display(len(data))

    f_data = pd.DataFrame(data, columns=['Attrition', 'DailyRate', 'EducationField', 'YearsAtCompany'
        , 'YearsInCurrentRole',	'YearsSinceLastPromotion',	'YearsWithCurrManager'
    ])

    # display(f_data.head())

    m_data = f_data[f_data['Attrition'] == 'Yes']
    f_data = m_data.drop( ['Attrition'], axis=1)

    # display(len(f_data))
    #
    # return 0

    # display(f_data.head())
    # return 0
    # display(data.shape)
    # X = f_data
    # y = f_data['Gender']
    # le = LabelEncoder()
    # X['Gender'] = le.fit_transform(X['Gender'])
    # y = le.transform(y)
    #
    # X = f_data
    # y = f_data['Department']
    # le = LabelEncoder()
    # X['Department'] = le.fit_transform(X['Department'])
    #
    # y = le.transform(y)

    # display(f_data['EducationField'].unique())
    X = f_data
    y = f_data['EducationField']
    le = LabelEncoder()
    X['EducationField'] = le.fit_transform(X['EducationField'])
    y = le.transform(y)
    # display(y)

    cols = X.columns

    ms = MinMaxScaler()
    X = ms.fit_transform(X)

    X = pd.DataFrame(X, columns=[cols])

    # display(X.head())


    # cs = []
    # for i in range(1, 12):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(X)
    #     cs.append(kmeans.inertia_)
    # plt.plot(range(1, 12), cs)
    # plt.title('The Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('CS')
    # plt.show()

    # calculate_k_value(X)
    # return 0

    find_k_means(X, y, 2)

    return 0


def calculate_k_value(x):
    error = []
    for i in range(1, 12):
        kmeans = KMeans(n_clusters=i).fit(x)
        kmeans.fit(x)
        error.append(kmeans.inertia_)
    import matplotlib.pyplot as plt
    plt.plot(range(1, 12), error)
    plt.title('Elbow method')
    plt.xlabel('No of clusters')
    plt.ylabel('Error')
    plt.show()


def find_k_means(x, y, k):
    k_means = KMeans(n_clusters=3, random_state=0)
    y_k_means = k_means.fit_predict(x)
    labels = k_means.labels_
    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    print('Accuracy score: {0:0.2f} %'.format((correct_labels *100)/ float(y.size)))
    # display(x.shape)

    plt.scatter(x['YearsAtCompany'],x['YearsWithCurrManager'], c=y_k_means, cmap='rainbow')
    plt.title('Scatter Plot with K = 3')
    plt.xlabel('Number of Years at Company')
    plt.ylabel('Years With Current Manager')
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


if __name__ == '__main__':
    # print_hi('PyCharm')
    _recommend()
