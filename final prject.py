#team:
# ziad elsayed ebrahim
# ziad sayed mohamed
# ziad sherief mohsen
# abdelrahman anwar mostafa
# omar hany saber
#-----------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
data = pd.read_csv(r"https://storage.googleapis.com/kagglesdsdata/datasets/3321433/5781088/Sleep_health_and_lifestyle_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230529T200444Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=93ba05502e8ac9b9765038ea124a52544b54ab6556ea6c3b6e0f5a128dbe92c183061578bf0a33c66ec1d8822838bbc99633e8ee838bfe03cb17f9f8d8f101f9ba8f0801315c6d692a43fd0cbb5bb3e716fb1f8081265ce6f0530e578e886ec76d4a1a0c7dddbdb31a40fbde96f4f23259be62cc5ab48c9fb1dbcb9f1b47af70053d81ce4b45501b3d048e67e33d13c2c3a9e3e001709cc55461759195c21dfa13614eb2e7d6ac3e42e9e5dd42c1ca315a7c2c05ca9034d9ffadabf12ffbaada1b600154925e17c7e6b68e9b07162e705aa7f6db6c878485efad74ae8c268d587b1ef991357f93392c60169eaafe7e79fa5cc0a31bf6a1bbb52110236ba4dfee")

#-------------------------------------------------------------------------------------

# Discover the data:
# print(data) # to show the data
# print(data.shape) # to show numbers of columns and raws
# print(data.head(0)) # to show the heads of columns
# print(data.dtypes) # to show the data types
# print(data.info()) # to know any information about data

#-------------------------------------------------------------------------------------
# data cleaning:
# print(data.isnull().sum()) # To see how many ”NaN” in each column
# print(data.dropna) # to delete any raw contain "NaN" (our data don't contain any none so we don't need it)
# print(data.nunique()) # to see how many element without repeating

#-------------------------------------------------------------------------------------

# plot the data:
# sleep_duration = data['Sleep Duration']
# plt.hist(sleep_duration, bins=10, edgecolor='black')
# plt.xlabel('Sleep Duration')
# plt.ylabel('Frequency')
# plt.title('Distribution of Sleep Duration')
# plt.show()


#-------------------------------------------------------------------------------------
#  data Encoding:
# to convert data to numeric data
d_types=data.dtypes
for i in range(data.shape[1]):
    if d_types[i]=='object':
        Pr_data = preprocessing.LabelEncoder()
        data[data.columns[i]]=Pr_data.fit_transform(data[data.columns[i]])
#         print("Column index = ", i)
#         print(Pr_data.classes_)
# print(data)


#-------------------------------------------------------------------------------------
# data scalling:
# Data Scaling process is used to convert all the data to be in range between 0 and 1.
# scaler = preprocessing.MinMaxScaler()
# Scaled_data = scaler.fit_transform(data) 
# Scaled_data = pd.DataFrame(Scaled_data,columns=data.columns)
# print(Scaled_data)

#-------------------------------------------------------------------------------------

# data corelation:
# Correlation is the statistical analysis of the relationship or dependency between two variables
# r = Scaled_data.corr()
# print(r)
# sns.heatmap(data.corr() , cbar= True)
# plt.show()

# #----------------------------------------------------------------------------------------
# Let's go to learn the machine.
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# data=data.select_dtypes(exclude=["object"])
# X = data.iloc[:,:-1].values
# Y = data.iloc[:,-1].values

# x_train , x_test , y_train , y_test = train_test_split(X , Y , train_size= 0.7)
# # print(x_train)
# model = LinearRegression()
# model.fit(x_train , y_train)
# print("the Accuracy  to (train) in LinearRegression is: " , model.score(x_train , y_train)) # Accuracy 
# print("the Accuracy  to (test) in LinearRegression is: " ,model.score(x_test , y_test)) # Accuracy 
# ypred = model.predict(x_test)
# print(ypred)



# #-------------------------------------------
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# X = data.iloc[:,:-1].values
# Y = data.iloc[:,-1].values
# x_train , x_test , y_train , y_test = train_test_split(X , Y , train_size= 0.5)#50%
# model = DecisionTreeRegressor(max_depth= 5)
# model.fit(x_train , y_train)
# print("the Accuracy  to (train) in DecisionTree is: " ,model.score(x_train , y_train))
# print("the Accuracy  to (test) in DecisionTree is: " , model.score(x_test , y_test))

# #-------------------------------------------

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split
# x = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# x_train , x_test , y_train , y_test = train_test_split(x , y , train_size= 0.5) # 50%
# model = KNeighborsRegressor(n_neighbors=5)
# model.fit(x_train , y_train)
# print("the Accuracy  to (train) in KNeighbors is: " , model.score(x_train , y_train))
# print("the Accuracy  to (test) in KNeighbors is: " , model.score(x_test , y_test))

# #----------------------------------------------------------------------------------

# mean absolute error:
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
data = pd.read_csv(r"https://storage.googleapis.com/kagglesdsdata/datasets/3321433/5781088/Sleep_health_and_lifestyle_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230529T200444Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=93ba05502e8ac9b9765038ea124a52544b54ab6556ea6c3b6e0f5a128dbe92c183061578bf0a33c66ec1d8822838bbc99633e8ee838bfe03cb17f9f8d8f101f9ba8f0801315c6d692a43fd0cbb5bb3e716fb1f8081265ce6f0530e578e886ec76d4a1a0c7dddbdb31a40fbde96f4f23259be62cc5ab48c9fb1dbcb9f1b47af70053d81ce4b45501b3d048e67e33d13c2c3a9e3e001709cc55461759195c21dfa13614eb2e7d6ac3e42e9e5dd42c1ca315a7c2c05ca9034d9ffadabf12ffbaada1b600154925e17c7e6b68e9b07162e705aa7f6db6c878485efad74ae8c268d587b1ef991357f93392c60169eaafe7e79fa5cc0a31bf6a1bbb52110236ba4dfee")
data = data.dropna()
Pr_data = preprocessing.LabelEncoder()
data['Sleep Disorder'] = Pr_data.fit_transform(data['Sleep Disorder'])
print(data['Sleep Disorder'])
Ypred_con = np.array([1, 1, 2, 2, 1]).reshape(-1, 1)
print("Size after cleaning:", data.shape)
Continues_Data = data.select_dtypes(exclude=["object"])
print(Continues_Data)
Y = data.iloc[:, -1].values
num_repeats = len(Y) // len(Ypred_con)
remainder = len(Y) % len(Ypred_con)
Ypred_con = np.tile(Ypred_con, (num_repeats, 1))
Ypred_con = np.vstack((Ypred_con, Ypred_con[:remainder]))
MAE = mean_absolute_error(Y, Ypred_con)
print("Mean Absolute Error:", MAE)
