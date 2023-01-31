import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score

# methods
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Detection(object):

    def preprocessing(self, df_dataset):
        # country codes dataset
        df_country = pd.read_csv('datasets\iso_country_codes.csv').rename(columns={'English short name lower case': 'Country'})

        # adding a feature which iso alpha 3 code for countries
        countries = dict(zip(df_country['Country'], df_country['Alpha-3 code']))
        df_dataset['country'] = df_dataset['geo_loc']
        df_dataset['country'].replace(countries, inplace=True)

        # adding network type feature
        # Identifying the type of network [A, B, C]
        def network_type(ip):
            ip_str = ip.split(".")
            ip = [int(x) for x in ip_str]

            if ip[0] >= 0 and ip[0] <= 127:
                return (ip_str[0], "A")
            elif ip[0] >= 128 and ip[0] <= 191:
                return (".".join(ip_str[0:2]), "B")
            else:
                return (".".join(ip_str[0:3]), "C")


        df_dataset['Network'] = df_dataset['ip_add'].apply(lambda x: network_type(x))
        # df_dataset['net_part'], df_dataset['net_type'] = zip(*df_dataset.Network)
        _, df_dataset['net_type'] = zip(*df_dataset.Network)
        df_dataset.drop(columns=['Network'], inplace=True)
        
        # label attribute changes -> bad:Malicious, good:Benign
        df_dataset.label.replace({'bad': 'Malicious', 'good': 'Benign'}, inplace=True)

        # HTTPS column -> yes: HTTPS, no: HTTP
        df_dataset.https.replace({'yes' : 'HTTPS', 'no' : 'HTTP'}, inplace = True)

        # dropping columns
        df_dataset.drop(['index', 'geo_loc', 'url', 'ip_add'], axis=1, inplace=True)
        df_dataset.head()

        # Counting the Special Characters in the content
        def count_special(string):
            count = 0
            for char in string:
                if not(char.islower()) and not(char.isupper()) and not(char.isdigit()):
                    if char != ' ':
                        count += 1
            return count

        # Adding Feature that shows the Number of Special Character in the Content
        df_dataset['special_char'] = df_dataset['content'].apply(lambda x: count_special(x))

        # Length of the Content
        df_dataset['content_len'] = df_dataset['content'].apply(lambda x: len(x))
        df_dataset.drop(['content'], axis=1, inplace=True)
        
        # These are the categorical features that needs to be converted into numeric features for modelling 
        categorical = df_dataset.select_dtypes('object').columns.tolist()
        ls = [element for element in categorical if element not in ['label']]
        # This le_dict will save the Label Encoder Class so that the same Label Encoder instance can be used for the test dataset
        le_dict = {}

        for feature in ls:
            le = LabelEncoder()
            le_dict[feature] = le
            df_dataset[feature] = le.fit_transform(df_dataset[feature])

        df_dataset.label.replace({'Malicious' : 1, 'Benign' : 0}, inplace = True)

        # Scaling the in training data
        ss_dict = {}
        
        for feature in ['content_len', 'special_char', 'url_len', 'tld', 'js_len', 'js_obf_len', 'country']:
            ss = StandardScaler()
            ss_fit = ss.fit(df_dataset[feature].values.reshape(-1, 1))
            ss_dict[feature] = ss_fit
            d = ss_fit.transform(df_dataset[feature].values.reshape(-1, 1))
            df_dataset[feature] = pd.DataFrame(d, index = df_dataset.index, columns = [feature])
        
        return df_dataset

# ------------------------------------------------------------------------------------------------------------------------------------

    def run(self, user_test_size=0.3):
        # load the dataset and general information about the dataset
        global df_dataset
        df_dataset = pd.read_csv('datasets\small_dataset.csv').rename(columns={"Unnamed: 0": "index"})
        global original_dataset
        original_dataset = df_dataset.copy(deep=True)

        df_dataset = self.preprocessing(df_dataset)

        x = df_dataset.drop('label', axis=1)
        y = df_dataset.label

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=user_test_size)

        # Classificaation
        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machines': SVC()
        }
        results=pd.DataFrame(columns=['Methods', 'Accuracy in %', 'F1-score', 'Recall', 'Precision'])
        i = 0
        global model
        for method, func in classifiers.items():
            model = func
            func.fit(x_train,y_train)
            pred = func.predict(x_test)
            results.loc[i]= [method,
                                100*np.round(accuracy_score(y_test,pred),decimals=3),
                                round(f1_score(y_test,pred),2),
                                round(recall_score(y_test,pred),2),
                                round(precision_score(y_test,pred),2)]
            i += 1
        
        print(results)
        return results

# ------------------------------------------------------------------------------------------------------------------------------------

    def predict(self):
        random_label = random.randint(0, 1)
        df_sample = df_dataset[df_dataset.label==random_label].sample(1)
        df_test = df_sample.drop('label', axis=1) 
        pred = model.predict(df_test)
        print(df_sample)
        index = df_sample.index
        df_sample = original_dataset.iloc[index]
        
        return df_sample, pred