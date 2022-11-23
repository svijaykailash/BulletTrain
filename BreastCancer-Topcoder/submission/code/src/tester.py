print('Test Program Started')
import pandas as pd
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

print('Files used as follows\n')
def main():

    for x in sys.argv:
        print(x)

    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Testing input file is missing.")
        return 1
    
    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Solution file is missing.")
        return 1

    if len(sys.argv) < 4 or len(sys.argv[3]) == 0:
        print("Model file is missing.")
        return 1
    
    print('Testing started.')

    test_file = sys.argv[1]
    solution_file = sys.argv[2]
    model_file = sys.argv[3]

    df_test = pd.read_csv(test_file)
    test_file_ids = df_test['id']
    df_test = df_test.drop("id" , axis=1)

    with open(model_file, "rb") as file:
        model = pickle.load(file)
    
    pred = model.predict_proba(df_test)
    print('predicted values',pred)

    df_predict_prob = pd.DataFrame(pred, columns =['Prob-0','prediction'])
    submission_data = pd.DataFrame({'id': test_file_ids, 'prediction': df_predict_prob['prediction']})
    print(submission_data.info())
    submission_data.to_csv(solution_file, index=False)
    
    print('Testing finished.')

    return 0

if __name__ == "__main__":
    main()
