import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression
from sklearn.metrics import r2_score , mean_squared_error ,confusion_matrix,accuracy_score
import string as st
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',200)
df = pd.read_csv('D:\\test1\\Instant_Orange_Internship\\Instant_Final_Project\\New folder\\GamingStudy_data.csv' , encoding = 'ISO-8859-1')

df.drop(['S. No.' , 'Timestamp'] , axis = 1 , inplace = True)

df['Hours_streams'] = df['Hours']+df['streams']
df.drop(  ((df[df['Hours_streams'] > 116].index) | (df[df['Hours_streams']==0].index)),
                                             axis=0,inplace=True)

df.GADE.fillna(df.GADE.mode()[0] , inplace=True) #1

df.streams.fillna(int(df.streams.mean()) , inplace = True)
df.Hours.fillna(int(df.Hours.mean()) , inplace = True)

df.League = df.League.str.lower().str.strip()

df.loc[(df['whyplay']== 'having fun') ,'League'] =df.loc[(df['whyplay']== 'having fun') ,'League'].fillna('unranked')
df.League.fillna('gold' , inplace = True)


df["League"] =df["League"].str.extract(r'^([a-z]+)')


golds = ['g', 'gv', 'golden' ,'glod' ,'golld' ,'golf', 'goled', 'golderino' ,'giii']
df['League'] = df.League.replace(golds , 'gold')


silvers = [ 'silverii' , 's' , 'sliver' , 'siver' , 'silber' , 'sil' , 'silveriv']
df['League'] = df.League.replace(silvers , 'silver')

plats = [
    'platinium' , 'platnium' , 'platin' ,'pplatinum' ,'plarinum' ,'platium', 'p' ,'platine' ,
    'platinun' ,'platonum' ,'platnum', 'plata' ,'plantinum',
    'platinuim' ,'platunum', 'plantinum' ,'platunum' ,'platinumm' ,'platv' ,'platina' , 'plat' 
]
df['League'] = df.League.replace(plats , 'platinum')
print("cleanning")
bronzers = ['bronce' , 'b' , 'broze' ,'lowest' , 'wood', 'elohell'] 
df['League'] = df.League.replace(bronzers , 'bronze')


unranked = ['none' ,'na', 'not' ,'n' ,'promos' ,'provisional' ,'placements' , 'dont' , 'was', 'unraked',
            'havent', 'never', 'nope', 'no', 'noone', 'don', 'of', 'unrranked', 'new', 'what', 'unrank' ,
            'ranked', 'placement', 'unrankt' , 'non', 'unfranked' , 'promotion', 'idk',
            'unplaced', 'probably', 'provisionals', 'didnt' ,'unrakned' , 'unfinished' , 'just' , 'x' ,
            'promotions' , 'unseeded' , 'haven']
df['League'] = df.League.replace(unranked , 'unranked')


diamonds =  ['d', 'dia', 'diaomnd', 'diamont','diamomd']
df['League'] = df.League.replace(diamonds , 'diamond')

chall =  ['challenjour', 'c', 'charrenjour', 'challeneger']
df['League'] = df.League.replace(chall , 'challenger')
print("cleanning2")

gm =  ['grand', 'gm', 'grandmasters']
df['League'] = df.League.replace(gm , 'grandmaster')

df['League'] = df.League.replace('mg' , 'mge')
df['League'] = df.League.replace('masters' , 'master')
df['League'] = df.League.replace( ['le', 'legdendary'] , 'legendary')

counts = df['League'].value_counts()
df['League'] = df['League'][~df['League'].isin(counts[counts < 3].index)]


df['League'] = df.League.replace(['i' , 'currently' , 'high' , 'season' , 'lol','cs' ,
                                  'last' ,'csgo','starcraft' ,'geater' , 'in', 'rank' , 'still'] , np.nan)

df.League.fillna('unspecified' , inplace=True)

df.whyplay = df.whyplay.str.lower().str.strip()
df['Narcissism'].fillna(df['Narcissism'].mode()[0],inplace=True)
df.drop(["Birthplace","Birthplace_ISO3"],axis=1,inplace=True)
print("cleanning3")

df['Residence'] = df['Residence'].replace('Unknown',df['Residence'].mode()[0])

df['Reference'].fillna('Other',inplace=True)

df.drop(df[df['accept'].isnull()].index , axis=0 , inplace=True)


df['Residence_ISO3'].fillna('USA',inplace=True) #11063

df.loc[11063,'Residence_ISO3'] = 'XXK'
col = ['SPIN1','SPIN2','SPIN3','SPIN4','SPIN5','SPIN6','SPIN7','SPIN8','SPIN9',
     'SPIN10','SPIN11','SPIN12','SPIN13','SPIN14','SPIN15','SPIN16','SPIN17' ,'SPIN_T']
for i in col :
    df[i].fillna(df[i].mode()[0], inplace = True)
    
df['Playstyle'] = df['Playstyle'].apply(lambda x: ' '.join(word.strip(st.punctuation) for word in x.split()))
df['earnings'] = df['earnings'].apply(lambda x: ' '.join(word.strip(st.punctuation) for word in x.split()))
df['whyplay'] = df['whyplay'].apply(lambda x: ' '.join(word.strip(st.punctuation) for word in x.split()))

df['Playstyle'] = df['Playstyle'].str.lower().str.strip()
df['whyplay'] = df['whyplay'].str.lower().str.strip()
df['earnings'] = df['earnings'].str.lower().str.strip()
print("cleanning4")
df.drop('highestleague' , axis = 1 , inplace = True)

df['Hours_streams'].fillna(df['Hours_streams'].median(),inplace=True)
df.Work.fillna(df.Work.mode()[0] , inplace=True)

df.drop(['Residence' , 'accept'] , axis = 1 , inplace = True)

df.earnings.replace(df.earnings.value_counts().index[3:] , 'Other',inplace=True)

df.whyplay.replace(df.whyplay.value_counts().index[5:] , 'Other',inplace=True)

df.Playstyle.replace(df.Playstyle.value_counts().index[5:] , 'Other',inplace=True)
df.drop([12117,10622] , axis = 0 , inplace=True)

df.drop( df[df['Hours_streams'] > 80].index , axis = 0 , inplace=True)
df.drop( df[df['Age'] > 35].index , axis = 0 , inplace=True)

print("cleanning before label encoding")
le = LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = le.fit_transform(df[i])
        
df = df[['GAD_T' , 'SWL_T' , 'SPIN_T' , 'Narcissism' , 'Age' ]]

pc = PCA(n_components=3)

x = pc.fit_transform(df)



model = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, n_init = 25, random_state = 0)
y_clusters = model.fit_predict(x)

df['Label'] = y_clusters


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train , X_test ,y_train , y_test = train_test_split(X,y,train_size=.8,random_state=44)

print("cleanning before randomforest")
clf = RandomForestClassifier(n_estimators = 500 , random_state=30, max_depth=8) 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print(clf.score(X_test,y_test))
import pickle
pickle.dump(le,open("D:\\test1\\Instant_Orange_Internship\\Instant_Final_Project\\New folder\\encoder.pkl",'wb'))
pickle.dump(clf,open("D:\\test1\\Instant_Orange_Internship\\Instant_Final_Project\\New folder\\RandomForestClass.pkl",'wb'))




