import numpy as np
import pandas as pd

rdat=pd.read_csv('train.csv')

dat=rdat[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
dat=dat.dropna(how="any")
dat=dat.replace({"male":0,"female":1,"Q":0,"S":1,"C":2})
print dat

npdat=dat.as_matrix()
print npdat
