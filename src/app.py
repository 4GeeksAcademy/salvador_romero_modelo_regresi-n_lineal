# your code here
def eliminar_atipicos(datos, columnas):
    new_data = datos
    for i in columnas:
        q1=new_data[i].quantile(0.25)
        q3=new_data[i].quantile(0.75)
        iqr = q3-q1
        low_lim = q1 - 1.5*iqr
        hi_lim = q3 + 1.5*iqr
        rem = new_data[(new_data[i]>=hi_lim) | (new_data[i]< low_lim)]
        new_data = new_data.drop(index=rem.index)
    return new_data.copy()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv")
data_uni = data.drop_duplicates()

col_obj = data_uni.select_dtypes(include='object').columns.tolist()
col_num = data_uni.select_dtypes(exclude='object').columns.tolist()

data_num_clean = eliminar_atipicos(data_uni,col_num)

# Escalar las columnas
scaler = StandardScaler()
norm_features = scaler.fit_transform(data_uni[col_num])
data_scal = pd.DataFrame(norm_features, index = data_uni.index, columns = col_num)
data_scal["Heart disease_number"] = data_uni["Heart disease_number"]

# Filtrar columnas 
correlation_matrix = data_scal.corr(numeric_only=True)
cor_target = correlation_matrix['Heart disease_number'].abs().sort_values(ascending=False)
top_features = cor_target[cor_target > 0.5]  # puedes cambiar el umbral
print(top_features[:20])
data_top = data_scal[top_features[:20].index]
X = data_top.drop(columns=["Heart disease_number"])
Y = data_scal["Heart disease_number"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=8)

# Regresión logística
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
rl_score = r2_score(y_test,y_pred)

# Modelo optimizado con Lasso
las_model = Lasso()
las_model.fit(x_train,y_train)
y_pred_las = las_model.predict(x_test)
las_score = r2_score(y_test,y_pred_las)

print(" Puntuación RL: "+ str(rl_score)+"\n Puntuación Lasso: "+str(las_score))