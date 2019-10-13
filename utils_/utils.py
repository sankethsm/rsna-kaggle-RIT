# We will only need OS and Pandas for this one
import os
import pandas as pd

def prepareDataframe(path, train=False, nrows=None):
    """
    Prepare Pandas DataFrame for fitting neural network models
    Returns a Dataframe with two columns
    ImageID and Labels (list of all labels for an image)
    """ 
    df = pd.read_csv(path, nrows=nrows)
    if train:
        # Duplicates found from this kernel:
        # https://www.kaggle.com/akensert/resnet50-keras-baseline-model
        removeDuplicates = [1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
                                312468,  312469,  312470,  312471,  312472,  312473,
                                2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
                                3032994, 3032995, 3032996, 3032997, 3032998, 3032999]  
        df = df.drop(index=removeDuplicates).reset_index(drop=True)
    
    # Get ImageID for using with generator
    df['ImageID'] = df['ID'].str.rsplit('_', 1).map(lambda x: x[0]) + '.png'
    # Get labels for each image
    labelList = df.groupby('ImageID')['Label'].apply(list)
    
    # A clean DataFrame with a column for ImageID and columns for each label
    newDf = pd.DataFrame({'ImageID': df['ImageID'].unique(), 
                           'Labels': labelList}).set_index('ImageID').reset_index()
    newDf[targets] = pd.DataFrame(newDf['Labels'].values.tolist(), index= newDf.index)
    newDf = newDf.drop('Labels', axis=1)
    return newDf

# Path names
rootFolder = "G:\\My Drive\\Kaggle"
trainSheet = os.path.join(rootFolder, 'stage_1_train.csv')
validSheet = os.path.join(rootFolder, 'stage_1_sample_submission.csv')

# All labels that we have to predict in this competition
targets = ['epidural', 'intraparenchymal', 
           'intraventricular', 'subarachnoid', 
           'subdural', 'any']

