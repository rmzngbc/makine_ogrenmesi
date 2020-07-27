import warnings
warnings.filterwarnings("ignore")


from PyQt5.QtWidgets import *
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
class Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.left=50
        self.top=50
        self.width=1080
        self.height=640
        self.title="Veri Bilimi"
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)
        self.tabWidget()
        self.widgets()
        self.layouts()
        self.show()
    
    #tab oluşturma
    def tabWidget(self):
        self.tabs=QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tab1=QWidget()
        self.tab2=QWidget()
        self.tab3=QWidget()
        self.tabs.addTab(self.tab1,"Sınıflandırma")
        self.tabs.addTab(self.tab2,"Doğrusal Regresyon")
        self.tabs.addTab(self.tab3,"Doğrusal Olmayan Regresyon")
    def widgets(self):
        #plot
        self.p=PlotCanvas(self,width=4,height=4)
        #label k
        self.k_number_text=QLabel("Makine Öğrenmesi algoritmasını seçiniz:")
        #radio buttonlar-makine öğrenmesi alogirtmalarının adları:
        self.mak_predict1=QRadioButton("K-En Yakın Komşu(KNN)")
        self.mak_predict2=QRadioButton("Yapay Sinir Ağları(YSA)")
        self.mak_predict3=QRadioButton("Lojistik Regresyon")
        self.mak_predict4=QRadioButton("Naive Bayes")
        self.mak_predict5=QRadioButton("Destek Vektör Sınıflandırıcısı(SVC)")
        self.mak_predict6=QRadioButton("Doğrusal Olmayan Destek Vektör Sınıflandırıcısı(SVC)")
        self.mak_predict7=QRadioButton("Classification and Regression Tress(CART)")
        self.mak_predict8=QRadioButton("Random Forest Sınıflandırma(RF)")
        self.mak_predict9=QRadioButton("Gradient Boosting Machines(GBM)")
        self.mak_predict10=QRadioButton("eXtereme Gradient Boosting(xGBoost)")
        self.mak_predict11=QRadioButton("Light GBM")
        self.mak_predict12=QRadioButton("Category Boosting(CatBoost)")
    
        #predict butonu
        self.predict=QPushButton("Model Kur",self)
        self.predict.clicked.connect(self.predictFunction)
        
        #dosya açma butonu
        self.file=QPushButton("Lütfen veri setini seçiniz",self)
        self.file.clicked.connect(self.fileFunction)
        
        #uyarı butonu
        self.uyari=QPushButton("Lütfen Dikkat!",self)
        self.uyari.clicked.connect(self.uyariFunction)
        
        ##tab2##!!!##################################
        
        #plot
        self.p2=PlotCanvas(self,width=4,height=4)
        #label k
        self.k_number_text2=QLabel("Makine Öğrenmesi algoritmasını seçiniz:")
        #radio buttonlar-makine öğrenmesi alogirtmalarının adları:
    
        self.mak_predict02=QRadioButton("Çoklu Doğrusal Regresyon")
        self.mak_predict03=QRadioButton("Temel Bileşen Regresyonu(PCR)")
        self.mak_predict04=QRadioButton("Kısmi En Küçük Kareler Regresyonu(PLS)")
        self.mak_predict05=QRadioButton("Ridge Regresyon")
        self.mak_predict06=QRadioButton("Lasso Regresyon")
        self.mak_predict07=QRadioButton("ElasticNet Regresyonu")
        
         #uyarı butonu
        self.uyari2=QPushButton("Lütfen Dikkat!",self)
        self.uyari2.clicked.connect(self.uyariFunction2)
        
        #predict butonu
        self.predict2=QPushButton("Model Kur",self)
        self.predict2.clicked.connect(self.predictFunction2)
        
        #dosya açma butonu
        self.file2=QPushButton("Lütfen veri setini seçiniz",self)
        self.file2.clicked.connect(self.fileFunction2)
        
        
        ##tab3:::
        
        #plot
        self.p3=PlotCanvas(self,width=4,height=4)
        #label k
        self.k_number_text3=QLabel("Makine Öğrenmesi algoritmasını seçiniz:")
        #radio buttonlar-makine öğrenmesi alogirtmalarının adları:
        self.mak_predict30=QRadioButton("K-En Yakın Komşu(KNN)")
        self.mak_predict31=QRadioButton("Doğrusal Olmayan Destek Vektör Regresyon(SVR)")
        self.mak_predict32=QRadioButton("Yapay Sinir Ağları(YSA)")
        self.mak_predict33=QRadioButton("Classification and Regression Tress(CART)")
        self.mak_predict34=QRadioButton("Bagged Trees")
        self.mak_predict35=QRadioButton("Random Forest(RF)")
        self.mak_predict36=QRadioButton("Gradient Boosting Machines(GBM)")
        self.mak_predict37=QRadioButton("eXtreme Gradient Boosting(XGBoost)")
        self.mak_predict38=QRadioButton("Light GBM ")
        self.mak_predict39=QRadioButton("Category Boosting(CatBoost)")
        
        
         #uyarı butonu
        self.uyari3=QPushButton("Lütfen Dikkat!",self)
        self.uyari3.clicked.connect(self.uyariFunction3)
        
        #predict butonu
        self.predict3=QPushButton("Model Kur",self)
        self.predict3.clicked.connect(self.predictFunction3)
        
        #dosya açma butonu
        self.file3=QPushButton("Lütfen veri setini seçiniz",self)
        self.file3.clicked.connect(self.fileFunction3)
        
        
        
       
    
    def KNN(self):
        
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
    
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        knn_params_df={'n_neighbors':np.arange(1,10)}
        knn=KNeighborsClassifier()   #model nesnesi
        knn_cv_df=GridSearchCV(knn,       #otpimum komluluk sayısını bulma
                        knn_params_df,
                        cv=10).fit(X_train,y_train)
        knn_cv_df.best_params_
        #final model kurma

        knn_tuned_df=KNeighborsClassifier(n_neighbors=knn_cv_df.best_params_['n_neighbors']).fit(X_train,y_train)
        y_pred2=knn_tuned_df.predict(X_test)
        
        y_pred2      #çıkan tahminler
        
        self.başarı1=accuracy_score(y_test,y_pred2)  #test hatası
        
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"KNN Modeli","Model Kuruldu!")
        
        #grafik-barhplot
        self.p.plot(["KNN"],self.başarı1)
               
        
       
    def YSA(self):
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
    
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.fit_transform(X_test)
        
        mlpc_params_df={"alpha":[0.1,0.01],
                 "hidden_layer_sizes":[(5,10),(10,10)],
                 "solver":["sgd","adam"],
                 "activation":["relu","logistic"]}
        mlpc=MLPClassifier()
        mlpc_cv_df=GridSearchCV(mlpc,
                         mlpc_params_df,
                         cv=10,
                         n_jobs=-1,
                         verbose=2)
        mlpc_cv_df.fit(X_train_scaled,y_train)
        
        mlpc_tuned_df=MLPClassifier(activation=mlpc_cv_df.best_params_['activation'],
                             alpha=mlpc_cv_df.best_params_['alpha'],
                             hidden_layer_sizes=mlpc_cv_df.best_params_['hidden_layer_sizes'],
                             solver=mlpc_cv_df.best_params_['solver']).fit(X_train_scaled,y_train)
        
        y_pred=mlpc_tuned_df.predict(X_test_scaled)
        y_pred
        #başarı skoru
        self.başarı2=accuracy_score(y_test,y_pred)
        #barhplot grafik
        self.p.plot(["YSA"],self.başarı2)
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"YSA Modeli","Model Kuruldu!")
       
        
    def Lojistik_Regresyon(self):
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
       
        
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #lojistikregresyon modeli kuruldu
        loj=LogisticRegression(solver="liblinear")
        loj_model_df=loj.fit(X_train,y_train)
        
        #tahmin işlemi
        y_pred=loj_model_df.predict(X_test)
        
        #başarı skoru
        self.başarı3=accuracy_score(y_test,y_pred)
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"LR Modeli","Model Kuruldu!")
        
        #barhplot grafik
        self.p.plot(["L.Regresyon"],self.başarı3)
       
        
        
        
        
    def Naive_Bayes(self):
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        nb=GaussianNB()  #model nesnesi
        nb_model_df=nb.fit(X_train,y_train)
        
        #tahmin
        y_pred=nb_model_df.predict(X_test)
        #skor
        self.başarı4=cross_val_score(nb_model_df,X_test,y_test,cv=10).mean()
        
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"Naive Bayes Modeli","Model Kuruldu!")
        
        #barhplot grafik
        self.p.plot(["Naive Bayes"],self.başarı4)
        
       
       
        
        
        
    def SVC(self):
        
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #MODEL-parametre belirleme
        svc_params={"C":np.arange(1,3)}
        svc=SVC(kernel="linear")  #model nesnemiz:
        svc_cv_model=GridSearchCV(svc,
                         svc_params,
                         cv=10,
                         n_jobs=-1,
                         verbose=2).fit(X_train,y_train)
        svc_cv_model.best_params_
        
        #final model
        svc_tuned_df=SVC(kernel="linear",C=svc_cv_model.best_params_['C']).fit(X_train,y_train)
        #tahmin
        y_pred=svc_tuned_df.predict(X_test)
        #başarı skoru:
        self.başarı5=accuracy_score(y_test,y_pred)
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"SVC Modeli","Model Kuruldu!")
        
        #barh-plot
        
        self.p.plot(["svc"],self.başarı5)
    
    
        
        
        
    def SVC_RBF(self):
        
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #parametre ayarı C ve gama
        svc_params={"C":[0.1,0.001],
           "gamma":[0.1,2,5]}
        svc=SVC() #model nesnemiz
        svc_cv_model=GridSearchCV(svc,svc_params,cv=10,n_jobs=-1,verbose=2)
        svc_cv_model.fit(X_train,y_train)
        svc_cv_model.best_params_
        
        
        #final modelimizi kuralım:
        svc_tuned_df=SVC(C=svc_cv_model.best_params_['C'],gamma=svc_cv_model.best_params_['gamma']).fit(X_train,y_train)
        #tahmin hesabı:
        y_pred=svc_tuned_df.predict(X_test)
        #test hatamız:
        self.başarı6=accuracy_score(y_test,y_pred)
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"SVC RBF Modeli","Model Kuruldu!")
        
        #barh-plot
        self.p.plot(["SVC-RBF"],self.başarı6)
        
       
        
        
        
        
    def CART(self):
        
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #optimum parametre seçme
        cart_grid={'max_depth':range(1,5),
          'min_samples_split':list(range(2,6))}
        
        
        #cv işlemi(optimum parametre için)
        cart=DecisionTreeClassifier()
        cart_cv=GridSearchCV(cart,
                    cart_grid,
                    cv=10,
                    n_jobs=-1,
                    verbose=2).fit(X_train,y_train)
        
        #final model
        cart_tuned=DecisionTreeClassifier(max_depth=cart_cv.best_params_['max_depth'],min_samples_split=cart_cv.best_params_['min_samples_split']).fit(X_train,y_train)
        #tahmin hesabı
        y_pred=cart_tuned.predict(X_test)
        
        #başarı skoru
        self.başarı7=accuracy_score(y_test,y_pred)
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"CART Modeli","Model Kuruldu!")
        #sonuç:
        
        #barh-plot
        self.p.plot(["CART"],self.başarı7)
    
        
        
        
        
        
        
        
    def RF(self):
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #parametre seçme
        rf_params={'max_depth':[2,5],
          'n_estimators':[10,100],
          'min_samples_split':[2,10]}
        
        #optmimum parametreyi bulma
        rf=RandomForestClassifier()
        rf_cv_model=GridSearchCV(rf,
                        rf_params,
                        cv=10,
                        n_jobs=-1,
                        verbose=2).fit(X_train,y_train)
        
        #final model:
        rf_tuned=RandomForestClassifier(max_depth=rf_cv_model.best_params_['max_depth'],
                                
                                        n_estimators=rf_cv_model.best_params_['n_estimators'],
                                        min_samples_split=rf_cv_model.best_params_['min_samples_split']).fit(X_train,y_train)
        #tahmin hesabı
        y_pred=rf_tuned.predict(X_test)
        #başarı skoru
        self.başarı8=accuracy_score(y_test,y_pred)
         #bilgi mesajı:
        m_box=QMessageBox.information(self,"RF Modeli","Model Kuruldu!")
        #barh-plot
        self.p.plot(["RF"],self.başarı8)
       
        
        
        
    def GBM(self):
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #parametre seçme:
        gbm_params={'learning_rate':[0.01,0.001],
           'n_estimators':[5,15],
           'max_depth':[3,5],
           'min_samples_split':[2,5]}
        #model kurarak optimum parametreyi bulma:
        gbm=GradientBoostingClassifier()  #model nesnemiz
        gbm_cv=GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)
        gbm_cv.fit(X_train,y_train)
    
        #final model:
        gbm_tuned=GradientBoostingClassifier(learning_rate=gbm_cv.best_params_['learning_rate'],
                                             max_depth=gbm_cv.best_params_['max_depth'],
                                             min_samples_split=gbm_cv.best_params_['min_samples_split'],
                                             n_estimators=gbm_cv.best_params_['n_estimators']).fit(X_train,y_train)
        #tahmin hesabı
        y_pred=gbm_tuned.predict(X_test)
        #başarı skoru
        self.başarı9=accuracy_score(y_test,y_pred)
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"RF Modeli","Model Kuruldu!")
        #barh-plot
        self.p.plot(["GBM"],self.başarı9)
       
    def XGBoost(self):
        
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #parametre seçme:
        xgb_params={'n_estimators':[5,10],
           'subsample':[0.6,0.8],
           'max_depth':[3,4],
           'learning_rate':[0.1,0.01],
           'min_samples_split':[2,5]}
        
        xgb=XGBClassifier() #model nesnesi
        #crsoss valide işlemi
        xgb_cv=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
        
        
        #final model:
        xgb_tuned=XGBClassifier(learning_rate=xgb_cv.best_params_['learning_rate'],
                                
                                max_depth=xgb_cv.best_params_['max_depth'],
                                
                                min_samples_split=xgb_cv.best_params_['min_samples_split'],
                                
                                n_estimators=xgb_cv.best_params_['n_estimators'],
                                
                                subsample=xgb_cv.best_params_['subsample']).fit(X_train,y_train)
        #tahmin hesabı
        y_pred=xgb_tuned.predict(X_test)
        #başarı sokur
        self.başarı10=accuracy_score(y_test,y_pred)
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"XGBoost Modeli","Model Kuruldu!")
        #barh-plot
        self.p.plot(["XGBM"],self.başarı10)
      
    def LightGBM(self):
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #parametre seçme:
        lgbm_params = {
        'n_estimators': [5,10],
        'subsample': [0.6,1.0],
        'max_depth': [3, 4],
        'learning_rate': [0.1,0.01],
        "min_child_samples": [5,10]}
        #optimum parametreleri bulma:
        
        lgbm = LGBMClassifier()   #model nesnesi:

        lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose = 2).fit(X_train,y_train)
        #final model:
        lgbm_tuned= LGBMClassifier(learning_rate = lgbm_cv_model.best_params_['learning_rate'], 
                       max_depth = lgbm_cv_model.best_params_['max_depth'],
                       subsample = lgbm_cv_model.best_params_['subsample'],
                       n_estimators = lgbm_cv_model.best_params_['n_estimators'],
                       min_child_samples = lgbm_cv_model.best_params_['min_child_samples']).fit(X_train,y_train)
        #tahmin hesabı:
        y_pred = lgbm_tuned.predict(X_test)
        
        #başarı skoru:
        self.başarı11=accuracy_score(y_test, y_pred)
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"LightGBM Modeli","Model Kuruldu!")
        #barh-plot
        self.p.plot(["LightGBM"],self.başarı11)
        
    def CatBoost(self):
        
        data=pd.read_csv(self.dosya_ismi[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
       
        
        #model kuruldu
        cat_model = CatBoostClassifier().fit(X_train, y_train)
        
        
        #tahmin hesabu
        y_pred = cat_model.predict(X_test)
        
        #başarı skoru
        self.başarı12=accuracy_score(y_test, y_pred)
        
        
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"CatBoost Modeli","Model Kuruldu!")
        #barh-plot
        self.p.plot(["CatBoost"],self.başarı12)
        
      
    
    
    def çdr(self):
        
        data=pd.read_csv(self.dosya_ismi2[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #model kurma
        lm=LinearRegression()
        model=lm.fit(X_train,y_train)
        #model tuning-optimum hatayı bulmc
        self.test_hatası=np.sqrt(-cross_val_score(model,
                                 X_test,
                                 y_test,
                                 cv=10,
                                 scoring="neg_mean_squared_error")).mean()
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"ÇDR Modeli","Model Kuruldu!")
        
        #grafik
        self.p2.plot2(["ÇDR"],self.test_hatası)
        
        
        
        
        
        
    def pcr(self):
        data=pd.read_csv(self.dosya_ismi2[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #model nesnesi ve değişken infirgeme(X_train ve X_test):
        pca=PCA()
        pca2=PCA()
        X_reduced_train=pca.fit_transform(scale(X_train))
        X_reduced_test=pca2.fit_transform(scale(X_test))
        #optimum bileşen sayısını bulma:
        cv_10=model_selection.KFold(n_splits=10,
                           shuffle=True,
                           random_state=1)
        
        lm=LinearRegression()
        RMSE=[]
        
        for i in np.arange(1, X_reduced_train.shape[1] + 1):
            score = np.sqrt(-1*model_selection.cross_val_score(lm, 
                                                       X_reduced_train[:,:i], 
                                                       y_train.ravel(), 
                                                       cv=cv_10, 
                                                       scoring='neg_mean_squared_error').mean())
            RMSE.append(score)
         #optimum bileşen sayısı   
        l=RMSE.index(min(RMSE))  
        #final model
        lm=LinearRegression()
        pcr_model=lm.fit(X_reduced_train[:,0:l],y_train)
        y_pred=pcr_model.predict(X_reduced_test[:,0:l])
        self.pcr_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
      
        
       
        #bilgi mesajı:
        m_box=QMessageBox.information(self,"PCR Modeli","Model Kuruldu!")
        
        #grafik:
        self.p2.plot2(["PCR"],self.pcr_test_hatası)
        
        
        
        
        
    def pls(self):
        data=pd.read_csv(self.dosya_ismi2[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #optimum bileşen sayısını bulma:
        #CV
        #10 katlı cross validaiton yöntemi kurduk.
    
        cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
        
        
        #Hata hesaplamak için döngü
        RMSE = []

        for i in np.arange(1, X_train.shape[1] + 1):
            pls = PLSRegression(n_components=i)
            score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
            RMSE.append(score)
        #optimum bileşen sayısı:
        l=RMSE.index(min(RMSE))
        
        #valide edilmiş final model:
        pls_model=PLSRegression(n_components=2).fit(X_train,y_train)
        y_pred=pls_model.predict(X_test)
        #laşabileceğimiz en optimum hata.
        self.pls_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        #uyarı mesajı:
        
        m_box=QMessageBox.information(self,"PLS Modeli","Model Kuruldu!")
        
        #grafik:
        self.p2.plot2(["PLS"],self.pls_test_hatası)
        
        
                
    def ridge(self):
        data=pd.read_csv(self.dosya_ismi2[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #optimum lambda değerini bulma:
        lambdalar=10**np.linspace(10,-2,100)*0.5
        ridge_cv=RidgeCV(alphas=lambdalar,
                scoring='neg_mean_squared_error',
                normalize=True)
        ridge_cv.fit(X_train,y_train)
        #optimum lambda değeri:
        ridge_cv.alpha_
        #final model:
        
        #tune edilmiş model
        ridge_tuned=Ridge(alpha=ridge_cv.alpha_,
                         normalize=True).fit(X_train,y_train)
        #final test hatası:
    
        self.ridge_test_hatası=np.sqrt(mean_squared_error(y_test,ridge_tuned.predict(X_test)))
        
        #uyarı mesajı
        m_box=QMessageBox.information(self,"Ridge Model","Model Kuruldu!")
        
        #grafik:
        self.p2.plot2(["Ridge"],self.ridge_test_hatası)
        
    def lasso(self):
        data=pd.read_csv(self.dosya_ismi2[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #optimum lambda değerini bulmak için cv işlemi
        lasso_cv_model=LassoCV(alphas=None,
                     cv=10,
                     max_iter=10000,
                     normalize=True)
        #model kuruldu:
        lasso_cv_model.fit(X_train,y_train)
        #cv modelin optimum alpha değeri
        lasso_cv_model.alpha_
        #final model:
        lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_)
        lasso_tuned.fit(X_train,y_train)
        #tahmin işlemi
        y_pred=lasso_tuned.predict(X_test)
        #final modelinin test hatası
        self.lasso_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        #uyarı mesajı
        m_box=QMessageBox.information(self,"Lasso Model","Model Kuruldu!")
        
        #grafik:
        self.p2.plot2(["Lasso"],self.lasso_test_hatası)
        
        
        
        
    def elastic(self):
        data=pd.read_csv(self.dosya_ismi2[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #cv işlemi optimum lambda değerini bulma
        enet_cv_model=ElasticNetCV(cv=10,random_state=0).fit(X_train,y_train)
        #optimum lambda değeri
        enet_cv_model.alpha_
        #final model:
        enet_tuned=ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train,y_train)
        #tahmin hesabı
        y_pred=enet_tuned.predict(X_test)
        #test hatası:
        self.elasticnet_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        #uyarı mesajı
        m_box=QMessageBox.information(self,"ElasticNet Model","Model Kuruldu!")
        
        #grafik:
        self.p2.plot2(["ElasticNet"],self.elasticnet_test_hatası)
        
        
      ##doğrusal olmayan regresyon:
      
    def knn_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #model nesnesi:
        knn=KNeighborsRegressor()
        #parametre aralığı(komşuluk sayısı):
        knn_params={'n_neighbors':np.arange(1,30,1)}
        #cv işlemi(girdsearh fonk.ile)
        knn_cv_model=GridSearchCV(knn,knn_params,cv=10).fit(X_train,y_train)
        #final model:
        knn_final_model=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_['n_neighbors']).fit(X_train,y_train)
        #tahmin hesabı:
        y_pred=knn_final_model.predict(X_test)
        #final test hatası:
        self.knn_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
        #uyarı mesajı
        m_box=QMessageBox.information(self,"KNN Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["KNN"],self.knn_test_hatası)
        
        
        
        
        
        
    def svr_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #model nesnsi:
        svr=SVR("rbf")
        #paramete seçme:
        svr_params={"C":[0.01,0.1,0.4,5,10,20,30,40,50]}
        #model tuning:
        svr_cv_model=GridSearchCV(svr,
                         svr_params,
                         cv=10).fit(X_train,y_train)
        #final model:
    
        svr_final_model=SVR("rbf",
             C=pd.Series(svr_cv_model.best_params_)[0]).fit(X_train,y_train)
        #tahmin hesabı:
        y_pred=svr_final_model.predict(X_test)
        #test hatası:
        self.svr_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        #uyarı mesajı
        m_box=QMessageBox.information(self,"SVR Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["SVR"],self.svr_test_hatası)
        
        
        
        
        
        
    def ysa_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #değişken dönüştürme işlemi:
        scaler=StandardScaler()
        scaler.fit(X_train)
        X_train_scaled=scaler.transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        #model nesnesi
        mlp_model=MLPRegressor()
        #parametre seçme:
    
        mlp_params={'alpha':[0.1,0.01],
           'hidden_layer_sizes':[(20,20),(10,30)],
           'activation':['relu','logistic']}
        #cv işlemi:
    
        mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=10).fit(X_train_scaled,y_train)
        #final model
        mlp_final_model=MLPRegressor(alpha=mlp_cv_model.best_params_['alpha'],
                      hidden_layer_sizes=mlp_cv_model.best_params_['hidden_layer_sizes']).fit(X_train_scaled,y_train)
        #tahmin hesabı:
        y_pred=mlp_final_model.predict(X_test_scaled)
        #test hatası:
        self.ysa_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
        #uyarı mesajı
        m_box=QMessageBox.information(self,"YSA Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["YSA"],self.ysa_test_hatası)
        
        
        
        
    def cart_r(self):
        
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #model nesnesi
        cart_model=DecisionTreeRegressor()
        #parametreler
        cart_params={"min_samples_split":range(2,100),
             "max_leaf_nodes":range(2,10)}
        
        #cv işlemi:
        cart_cv_model=GridSearchCV(cart_model,
                          cart_params,
                          cv=10).fit(X_train,y_train)
        #final model:
        
        cart_final_model=DecisionTreeRegressor(max_leaf_nodes=cart_cv_model.best_params_['max_leaf_nodes'],
                                         min_samples_split=cart_cv_model.best_params_['min_samples_split']).fit(X_train,y_train)
        #tahmin hesabı:
        y_pred=cart_final_model.predict(X_test)
        #hata hesabı:
        self.cart_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
        #uyarı mesajı
        m_box=QMessageBox.information(self,"CART Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["CART"],self.cart_test_hatası)
        
    def bagging_r(self):
        
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #model nesnesi
        bag_model=BaggingRegressor(bootstrap_features=True)
        #parametre ayarı
        bag_params={"n_estimators":range(2,20)}
        
        #optimum parametreyi bulma
        bag_cv_model=GridSearchCV(bag_model,
                         bag_params,
                         cv=10).fit(X_train,y_train)
        #final model
        bag_final_model=BaggingRegressor(n_estimators=bag_cv_model.best_params_['n_estimators'],random_state=45).fit(X_train,y_train)
        
        #tahmin işlemi:
        y_pred=bag_final_model.predict(X_test)
        #test hatası:
        self.bagging_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
         #uyarı mesajı
        m_box=QMessageBox.information(self,"Bagged Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["Bagged"],self.bagging_test_hatası)
        
        
        
        
        
        
    def rf_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #model nesnesi
        rf_model=RandomForestRegressor(random_state=42)
        #parametre seçme
        rf_params={'max_depth':list(range(1,5)),
          'max_features':[3,5],
          'n_estimators':[50,150]}
        #cv işlemi,best parametreler bulundu
        rf_cv_model=GridSearchCV(rf_model,
                        rf_params,
                        cv=10,
                        n_jobs=-1).fit(X_train,y_train)
        
        #final model:
        
        rf_final_model=RandomForestRegressor(max_depth=rf_cv_model.best_params_['max_depth'],
                                             max_features=rf_cv_model.best_params_['max_features'],
                                             n_estimators=rf_cv_model.best_params_['n_estimators']).fit(X_train,y_train)
        
        #tahmin hesabı:
        y_pred=rf_final_model.predict(X_test)
        #test hesabı
        self.rf_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
        #uyarı mesajı
        m_box=QMessageBox.information(self,"RF Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["RF"],self.rf_test_hatası)
        
        
        
        
        
    def gbm_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #model nesnesi:
        gbm=GradientBoostingRegressor()
        
        #parametre seçme:
        gbm_params={
            'learning_rate':[0.001,0.01],
            'max_depth':[3,5,8],
            'n_estimators':[50,100],
            'subsample':[1,0.5]
        }
        
        #cv işlemi-optimum parametreleri bulma:
        gbm_cv_model=GridSearchCV(gbm,
                                 gbm_params,
                                 cv=10,
                                 n_jobs=-1,
                                 verbose=2).fit(X_train,y_train)
        #final model:

        gbm_final_model=GradientBoostingRegressor(learning_rate=gbm_cv_model.best_params_['learning_rate'],
                                           max_depth=gbm_cv_model.best_params_['max_depth'],
                                           n_estimators=gbm_cv_model.best_params_['n_estimators'],
                                           subsample=gbm_cv_model.best_params_['subsample']).fit(X_train,y_train)
        #tahmin hesabı:
    
        y_pred=gbm_final_model.predict(X_test)
        
        #test hatası
        self.gbm_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
        #uyarı mesajı
        m_box=QMessageBox.information(self,"GBM Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["GBM"],self.gbm_test_hatası)
        
        
        
        
        
        
        
    def xgboost_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #model nesnesi
        xgb_model=XGBRegressor()
        #parameter aralıkları:
        
        xgb_grid={'colsample_bytree':[0.4,0.5],
                 'n_estimators':[50,100],
                 'max_depth':[2,3],
                 'learning_rate':[0.1,0.5]}
        #cv işlemi:
        xgb_cv_model=GridSearchCV(xgb_model,
                         param_grid=xgb_grid,
                         cv=10,
                         n_jobs=-1,
                         verbose=2).fit(X_train,y_train)
        #final model:
        
        xgb_final_model=XGBRegressor(colsample_bytree=xgb_cv_model.best_params_['colsample_bytree'],
                              learning_rate=xgb_cv_model.best_params_['learning_rate'],
                              max_depth=xgb_cv_model.best_params_['max_depth'],
                              n_estimators=xgb_cv_model.best_params_['n_estimators']).fit(X_train,y_train)
        #tahmin hesabı:
        y_pred=xgb_final_model.predict(X_test)

        #test hatası
        self.xgb_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        #uyarı mesajı
        m_box=QMessageBox.information(self,"XGB Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["XGB"],self.xgb_test_hatası)
        
    def lightgbm_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        #model nesnesi:
        lgbm=LGBMRegressor()
        #parametre belirleme:
        lgbm_grid={
            'colsample_bytree':[0.4,0.6],
            'learning_rate':[0.1,0.5],
            'n_estimators':[20,50],
            'max_depth':[1,2,3]
        }
        #cv işlemi
    
        lgbm_cv_model=GridSearchCV(lgbm,
                                  lgbm_grid,cv=10,
                                  n_jobs=-1,verbose=2).fit(X_train,y_train)
        #final model
        
        lgbm_final_model=LGBMRegressor(colsample_bytree=lgbm_cv_model.best_params_['colsample_bytree'],
                                learning_rate=lgbm_cv_model.best_params_['learning_rate'],
                                max_depth=lgbm_cv_model.best_params_['max_depth'],
                                n_estimators=lgbm_cv_model.best_params_['n_estimators']).fit(X_train,y_train)
        #tahmin
        y_pred=lgbm_final_model.predict(X_test)
        
        #test hatası
        self.lgbm_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
          #uyarı mesajı
        m_box=QMessageBox.information(self,"LGBM Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["LGBM"],self.lgbm_test_hatası)
        
        
    def catboost_r(self):
        data=pd.read_csv(self.dosya_ismi3[0])
        df=data.copy()
        df.head()
        
        X=X=df.iloc[::,:-1]   #bağımsız değişkenleri ayırma(dinamik)
        y=df.iloc[::,-1]              #bağımlı değişkeni y'ye atama(dinamik)
        
         #train-test işlemi:
        X_train,X_test,y_train,y_test=train_test_split(X,
                                              y,
                                              test_size=0.30,
                                              random_state=42)
        
        #model nesnesi
        catb_model=CatBoostRegressor().fit(X_train,y_train)
        
        
        #tahmin hesabı
    
        y_pred=catb_model.predict(X_test)
        #test hatası
        self.catb_test_hatası=np.sqrt(mean_squared_error(y_test,y_pred))
        
          #uyarı mesajı
        m_box=QMessageBox.information(self,"CatBoost Model","Model Kuruldu!")
        
        #grafik:
        self.p3.plot3(["CatBoost"],self.catb_test_hatası)
        
        
        
        
        
    
    #tab2
    #dosya açma fonskiyonu
    def fileFunction2(self):
        
        self.dosya_ismi2=QFileDialog.getOpenFileName(self,"Dosya Aç",os.getenv("HOME"))   #dosya açma işlemi #dosyanın dizini
        self.p2.clear()   #yeni bir veri seti seçildiğinde barh-plot silinecek
    
            
    
    
    
    
    def predictFunction2(self):
        
        if self.mak_predict02.isChecked():
            self.çdr()
        elif self.mak_predict03.isChecked():
            self.pcr()
        elif self.mak_predict04.isChecked():
            self.pls()
        elif self.mak_predict05.isChecked():
            self.ridge()
        elif self.mak_predict06.isChecked():
            self.lasso()
        elif self.mak_predict07.isChecked():
            self.elastic()
            
    def uyariFunction2(self):
        m_box=QMessageBox.information(self,"bağımlı değişken","Model kurmaya başlamadan önce veri setindeki bağımlı değişken son sütunda olmalıdır...")
            
    
        
            
        #tab1 
        #dosya açma fonskiyonu
    def fileFunction(self):
        
        self.dosya_ismi=QFileDialog.getOpenFileName(self,"Dosya Aç",os.getenv("HOME"))   #dosya açma işlemi #dosyanın dizini
        self.p.clear()   #yeni bir veri seti seçildiğinde barh-plot silinecek
    
    
    
    
    
    
    def predictFunction(self):
        
        if self.mak_predict1.isChecked():
            self.KNN()
        elif self.mak_predict2.isChecked():
            self.YSA()
        elif self.mak_predict3.isChecked():
            self.Lojistik_Regresyon()
        elif self.mak_predict4.isChecked():
            self.Naive_Bayes()
        elif self.mak_predict5.isChecked():
            self.SVC()
        elif self.mak_predict6.isChecked():
            self.SVC_RBF()
        elif self.mak_predict7.isChecked():
            self.CART()
        elif self.mak_predict8.isChecked():
            self.RF()
        elif self.mak_predict9.isChecked():
            self.GBM()
        elif self.mak_predict10.isChecked():
            self.XGBoost()
        elif self.mak_predict11.isChecked():
            self.LightGBM()
        elif self.mak_predict12.isChecked():
            self.CatBoost()
        
        
    def uyariFunction(self):
        m_box=QMessageBox.information(self,"bağımlı değişken","Model kurmaya başlamadan önce veri setindeki bağımlı değişken son sütunda olmalıdır...")
    
    def predictFunction3(self):
        if self.mak_predict30.isChecked():
            self.knn_r()
        elif self.mak_predict31.isChecked():
            self.svr_r()
        elif self.mak_predict32.isChecked():
            self.ysa_r()
        elif self.mak_predict33.isChecked():
            self.cart_r()
        elif self.mak_predict34.isChecked():
            self.bagging_r()
        elif self.mak_predict35.isChecked():
            self.rf_r()
        elif self.mak_predict36.isChecked():
            self.gbm_r()
        elif self.mak_predict37.isChecked():
            self.xgboost_r()
        elif self.mak_predict38.isChecked():
            self.lightgbm_r()
        elif self.mak_predict39.isChecked():
            self.catboost_r()
        
    def fileFunction3(self):
        self.dosya_ismi3=QFileDialog.getOpenFileName(self,"Dosya Aç",os.getenv("HOME"))   #dosya açma işlemi #dosyanın dizini
        self.p3.clear()   #yeni bir veri seti seçildiğinde barh-plot silinecek
    
    def uyariFunction3(self):
        m_box=QMessageBox.information(self,"bağımlı değişken","Model kurmaya başlamadan önce veri setindeki bağımlı değişken son sütunda olmalıdır...")
    
            
    #sayfa düzeni:
    def layouts(self):
        #layout belirle
        self.mainlayout=QHBoxLayout()
        self.leftlayout=QFormLayout()
        self.rightlayout=QFormLayout()
     #left
        self.leftlayoutGroupBox=QGroupBox("Veri Bilimi")
        self.leftlayout.addRow(self.k_number_text)
        self.leftlayout.addRow(self.mak_predict1)
        self.leftlayout.addRow(self.mak_predict2)
        self.leftlayout.addRow(self.mak_predict3)
        self.leftlayout.addRow(self.mak_predict4)
        self.leftlayout.addRow(self.mak_predict5)
        self.leftlayout.addRow(self.mak_predict6)
        self.leftlayout.addRow(self.mak_predict7)
        self.leftlayout.addRow(self.mak_predict8)
        self.leftlayout.addRow(self.mak_predict9)
        self.leftlayout.addRow(self.mak_predict10)
        self.leftlayout.addRow(self.mak_predict11)
        self.leftlayout.addRow(self.mak_predict12)
        self.leftlayout.addRow(self.uyari)
        self.leftlayout.addRow(self.file)
        self.leftlayout.addRow(self.predict)
       
        self.leftlayoutGroupBox.setLayout(self.leftlayout)
     #right
        self.rightlayoutGroupBox=QGroupBox("Başarı Skorları")
        self.rightlayout.addRow(self.p)
        self.rightlayoutGroupBox.setLayout(self.rightlayout)
    #main-->tab
        self.mainlayout.addWidget(self.leftlayoutGroupBox,50)
        self.mainlayout.addWidget(self.rightlayoutGroupBox,50)
        
        self.tab1.setLayout(self.mainlayout)
        
        ##tab2
         #layout belirle
        self.mainlayout2=QHBoxLayout()
        self.leftlayout2=QFormLayout()
        self.rightlayout2=QFormLayout()
        
        #left
        self.leftlayoutGroupBox2=QGroupBox("Veri Bilimi")
        self.leftlayout2.addRow(self.k_number_text2)
        
        self.leftlayout2.addRow(self.mak_predict02)
        self.leftlayout2.addRow(self.mak_predict03)
        self.leftlayout2.addRow(self.mak_predict04)
        self.leftlayout2.addRow(self.mak_predict05)
        self.leftlayout2.addRow(self.mak_predict06)
        self.leftlayout2.addRow(self.mak_predict07)
        self.leftlayout2.addRow(self.uyari2)
        self.leftlayout2.addRow(self.file2)
        self.leftlayout2.addRow(self.predict2)
       
        self.leftlayoutGroupBox2.setLayout(self.leftlayout2)
     #right
        self.rightlayoutGroupBox2=QGroupBox("Test Hataları")
        self.rightlayout2.addRow(self.p2)
        self.rightlayoutGroupBox2.setLayout(self.rightlayout2)
    #main-->tab
        self.mainlayout2.addWidget(self.leftlayoutGroupBox2,50)
        self.mainlayout2.addWidget(self.rightlayoutGroupBox2,50)
        
        self.tab2.setLayout(self.mainlayout2)
    #tab3
         #layout belirle
        self.mainlayout3=QHBoxLayout()
        self.leftlayout3=QFormLayout()
        self.rightlayout3=QFormLayout()
        
        #left
        self.leftlayoutGroupBox3=QGroupBox("Veri Bilimi")
        self.leftlayout3.addRow(self.k_number_text3)
        self.leftlayout3.addRow(self.mak_predict30)
        self.leftlayout3.addRow(self.mak_predict31)
        self.leftlayout3.addRow(self.mak_predict32)
        self.leftlayout3.addRow(self.mak_predict33)
        self.leftlayout3.addRow(self.mak_predict34)
        self.leftlayout3.addRow(self.mak_predict35)
        self.leftlayout3.addRow(self.mak_predict36)
        self.leftlayout3.addRow(self.mak_predict37)
        self.leftlayout3.addRow(self.mak_predict38)
        self.leftlayout3.addRow(self.mak_predict39)
        self.leftlayout3.addRow(self.uyari3)
        
        self.leftlayout3.addRow(self.file3)
        self.leftlayout3.addRow(self.predict3)
       
        self.leftlayoutGroupBox3.setLayout(self.leftlayout3)
     #right
        self.rightlayoutGroupBox3=QGroupBox("Test Hataları")
        self.rightlayout3.addRow(self.p3)
        self.rightlayoutGroupBox3.setLayout(self.rightlayout3)
    #main-->tab
        self.mainlayout3.addWidget(self.leftlayoutGroupBox3,50)
        self.mainlayout3.addWidget(self.rightlayoutGroupBox3,50)
        
        self.tab3.setLayout(self.mainlayout3)
         
     
        
        

class PlotCanvas(FigureCanvas):
    def __init__(self,parent=None,width=4,height=4,dpi=100):
        
        self.fig=Figure(figsize=(width,height),dpi=dpi)
        
        FigureCanvas.__init__(self,self.fig)
    def plot(self,x,y):
        
        
        self.ax=self.figure.add_subplot(111)
        self.ax.barh(x,y)
        self.ax.set_title("Veri Bilimi")
        self.ax.set_xlabel("Başarı skorları")
        self.ax.set_ylabel("Algoritmalar")
        self.ax.set_xticks(np.arange(0,1.1,0.1))
        self.draw()
    def plot2(self,x,y):
        
        self.ax=self.figure.add_subplot(111)
        self.ax.barh(x,y)
        self.ax.set_title("Veri Bilimi")
        self.ax.set_xlabel("Test Hataları")
        self.ax.set_ylabel("Algoritmalar")
        self.ax.set_xticks(np.arange(0,550,50))
        self.draw()
    
    def plot3(self,x,y):
        
        self.ax=self.figure.add_subplot(111)
        self.ax.barh(x,y)
        self.ax.set_title("Veri Bilimi")
        self.ax.set_xlabel("Test Hataları")
        self.ax.set_ylabel("Algoritmalar")
        self.ax.set_xticks(np.arange(0,550,50))
    
        
        self.draw()
    def clear(self):
        self.fig.clf()
        
       
        
    

    
    
    
    
    
    
w=Window()