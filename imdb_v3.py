import numpy as np
import pandas as pd

#ENVIRONMENT SETUP

#ortamımda hangi paketler mevcut
#conda list

#çalışmam için pc'de olması gereken paket ortamını base'den dosyalıyorum.
#conda env export > enviroment.yaml

#veriseti için yeni ortam oluşturma
#conda create -n zomato_env

#zomato virtual environment'ı aktive ediyorum
#conda activate tupras_env

#ham datayı içeren sheeti okuyorum
df = pd.read_csv('C:/Users/ybsoy/Desktop/IMDbMovies_dataset.csv')

#ilk 5 satırı okutuyorum
df.head()

#descriptive statistics incelemeleri yapıyorum



#sayısal değişkenler için ayrı bir tanımlama yapıyorum
num_var= [col for col in df.columns if df[col].dtype != 'O']
num_var



#temel descriptive fonksiyonlarını list'liyorum
desc_agg = ['sum', 'mean', 'std', 'var', 'min', 'max']
desc_agg

#bu fonksiyonları sayısal değerlere uyguluyorum
desc_agg_dict = {col: desc_agg for col in num_var}
desc_agg_dict

#değerleri tüm numeric variablelar için çalıştırıyorum.
desc_summ = df[num_var].agg(desc_agg_dict)
desc_summ
#desc_summ'ı print ediyorum, böylece her değişkenin toplam, standart sapma, min, max değerlerini incelemiş oluyorum.


#numpy array'a dönüştürmek istiyorum. Böylece değişkenin toplam, ort, std sapma vs değerlerini incelemiş oluyorum.
df_desc_na = np.array(desc_summ)
df_desc_na

#df numpy array olarak kullanılmak isterse;vektörel işlemler vb.
df_na = np.array(df)
df_na

#Overview devam

import seaborn as sns

df.shape
# (9083, 15)

df.info()
df.columns
#tüm dataları non-null MISSING DATA YOK, float, numeric gibi bilgileri gösterir.

#missing value için ekstra kontrol yapıyorum. TRUE geliyor.
df.isnull().values.any()

#her bir değişkene ait descriptive analytics değerlerini bir tabloya yeniden yazdırıyorum.
desc_summv2 = df.describe().T #yukarıda tek tek yaptığımız descriptive işlemini tek seferde yapmaya yarar bu fonksiyon.
df.info()

#Target'ı inceleyelim
#df[df.Target> df.Target.mean()].Target.count()
#df[df.Target < df.Target.mean()].Target.count()
#3      count       mean        std

count_above_mean = df[df['Budget (in millions)'] > df['Budget (in millions)'].mean()]['Budget (in millions)'].count()
print(count_above_mean) #993 tanesi ortalamadan büyük

count_under_mean = df[df['Budget (in millions)'] < df['Budget (in millions)'].mean()]['Budget (in millions)'].count()
print(count_under_mean) #4886 tanesi ortalamadan küçük


budget_data = df.loc[df['Budget (in millions)'] > df['Budget (in millions)'].mean(), ['Budget (in millions)']].head()
print(budget_data)
# "Budget (in millions)" sütunundaki değerlerin ortalamasından büyük olan satırları seçtik ve yalnızca "Budget (in millions)" sütununu içeren bir DataFrame döndüştürüp
# ilk 5 satırı head fonksiyonuyla aldık.


#orijinal DataFrame'den belirli bir sütun aralığını içeren yeni bir DataFrame oluşturduk.
# 5. sütundan 15 sütuna kadar olan aralığı aldık.
#herhangi bir kolonu hariç tutmak istiyorsan: imdb = df.iloc[:, [col for col in range(5, 15) if col != 13]]
#13'te tarihler var ve aşağıda kabul etmiyor numeric olarak.

imdb = df.iloc[:, [col for col in range(6, 15) if col != 13]]
imdb
imdb.columns


from matplotlib import pyplot as plt
import seaborn as sns

# Değişkenlerin grafiklerini çiziyorum
sns.boxplot(x=imdb['Budget (in millions)'])
plt.show()


def num_summary(imdb, columns=['Budget (in millions)'], plot=True):
    quantities = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(imdb[columns].describe(percentiles=quantities).T) #not: Pandas'ın describe fonksiyonu, quantile yerine 'percentiles' parametresini bekler.

    if plot:
        imdb[columns].hist()
        plt.xlabel(columns)
        plt.title(columns)
        plt.show(block=True)

num_summary(imdb, columns=['Budget (in millions)'], plot=True)

#tüm değişkenler için bir kod ile grafikler üretiyorum.
for col in imdb:                        # : bulunan her fonksiyon birlikte çalıştırılır.
    num_summary(imdb, col, plot=True)   # bütün değişkenlerimizi tek bir for döngüsüyle çalıştırmış olduk.

#bağımlı değişkenin bağımsız değişkenler üzerinden analiz ediiyorum.

df.groupby('Release Year')['Budget (in millions)'].mean()

#tüm sayısal değerler için bu fonksiyonu itere etmek istiyorum.

def imdb_summary_with_num(dataframe, imdb, column):
    print(dataframe.groupby(imdb).agg({column: 'mean' }), end='\n\n\n')

 # tüm sayısal kolonlar için çalışsın.
for col in imdb.columns:
    imdb_summary_with_num(df, ['Budget (in millions)'], col)

# tüm korelasyonları çıkarıyoruz.
corr = imdb.corr()
corr

# korelasyon ısı haritası çıkarıyoruz
sns.set(rc={'figure.figsize': (12, 12)})
plt.figure(figsize=(12, 12))
sns.heatmap(corr, cmap='RdBu') #RdBu: Red&Blue rengi
plt.show()


cor_matrix = df.corr().abs()
cor_matrix

#korelasyonda belli bir sınıfın altına kalanları drop etme işlemi gerçekleştiriyoruz.
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.80)]
cor_matrix[drop_list]


#Modeling
#Prediction
#Evaluation
#Hyperparameter Optimization
#Finalization

# Gerekli kütüphaneleri import edelim
#import matplotlib.pyplot as plt
#import yellowbrick as yb
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import MinMaxScaler
#from yellowbrick.cluster import KElbowVisualizer
#from scipy.cluster.hierarchy import linkage
#from scipy.cluster.hierarchy import dendrogram
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt  # Grafik çizimi için kütüphane
import yellowbrick as yb  # Yellowbrick, görselleştirmeler için kullanılan bir kütüphanedir
from sklearn.cluster import KMeans  # KMeans kümeleme algoritması için kütüphane
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Veri ölçekleme için kütüphaneler
from yellowbrick.cluster import KElbowVisualizer  # Küme sayısını belirlemek için görselleştirme aracı
from scipy.cluster.hierarchy import linkage, dendrogram  # Hiyerarşik kümeleme için kütüphaneler
from sklearn.decomposition import PCA  # Temel Bileşen Analizi için kütüphane
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için kütüphane
from sklearn.linear_model import LinearRegression  # Doğrusal Regresyon modeli için kütüphane
from sklearn.metrics import mean_squared_error, r2_score  # Regresyon modelinin performansını ölçmek için kütüphaneler


# Optimum küme sayısını belirliyoruz.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(X_scaled_v1)
elbow.show()

optimal_clusters = elbow.elbow_value_
print("Optimal number of clusters:", optimal_clusters)


# Veri setindeki eksik değerleri kontrol ediyoruz.
print(df.isnull().sum())

# Eksik değerleri 0 ile dolduruyoruz.
df.fillna(0, inplace=True)

X = df.drop(['Timestamp'], axis=1)




