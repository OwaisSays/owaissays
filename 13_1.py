from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv("Iris.csv")
kmm = KMeans(n_clusters=3) 
y_predicted = kmm.fit_predict(df[['Id','SepalLengthCm']])
print(y_predicted) 
df['cluster']=y_predicted 
print(km.cluster_centers_) 
