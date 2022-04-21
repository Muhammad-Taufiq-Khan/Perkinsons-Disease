from Function import *

PDvsRBD = "https://raw.githubusercontent.com/Muhammad-Taufiq-Khan/Perkinsons-Disease/main/Research%20Project%20(February%202022%20-%20September%202022)/Project%20Codes%20%26%20Data/Early%20Biomarkers%20of%20Parkinson's%20Disease/Modified%20PDvsRBD.csv"
data = pd.read_csv(PDvsRBD)

# Spliting Dataset
xtrain,xtest, ytrain, ytest = train_test_split(data.drop("Class", axis =1), data["Class"], test_size = 0.15, random_state=19)

#Preprocessing
xtrain, ytrain, xtest, ytest = Preprocessing(xtrain, ytrain, xtest, ytest)

#Cook Models
cvModels, models = CookModels(xtrain, ytrain)

# # Evaluation
# Evaluation(xtest, ytest, models)

Validation(xtrain, ytrain, cvModels)

