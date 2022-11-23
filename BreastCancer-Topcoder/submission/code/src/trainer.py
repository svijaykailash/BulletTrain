print('Train Program Started')
import pandas as pd
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')
import sklearn as sklearn
#from sklearn.model_selection import train_test_split
#from sklearn import metrics
#from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix,plot_confusion_matrix
import xgboost as xgboost
from xgboost import XGBClassifier

from platform import python_version
print(python_version())
print(pd.show_versions())
# print('sys version is {}.'.format(sys.version))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The xgb version is {}.'.format(xgboost.__version__))

print('Files used as follows\n')
def main():
    for x in sys.argv:
        print(x)

    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Training input file is missing.")
        return 1
    
    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Training output file (i.e Model file) is missing.")
        return 1
    
    print('Training started.')
    
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    
    df = pd.read_csv(input_file)
    df = df.drop("id" , axis=1)

    # Copy all the predictor variables into X dataframe
    X = df.drop(['cancer'],axis=1) 
    # Copy target into the y dataframe. 
    y = df[['cancer']]    
    
    # splitting data into training and test set for independent attributes
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1,stratify=df['cancer']) 
   
    model= XGBClassifier()
    model.fit(X, y)

    with open(model_file, "wb") as file:
        pickle.dump(model, file)

    print('Training finished.')

    return 0

if __name__ == "__main__":
    main()