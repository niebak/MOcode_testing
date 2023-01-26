from code_functions import *
from tabulate import tabulate
from scipy.fftpack import fft
from scipy.spatial.distance import cosine

def find_matching_trails(df, segments, percent_diff):
    # Create a subset of the dataframe that contains the segments within the range and percentage difference
    subset = df[(df['class'] == segments['class']) & 
                (df['curvature'] >= segments['curvature']*(1-percent_diff/100)) & (df['curvature'] <= segments['curvature']*(1+percent_diff/100)) &
                (df['climb'] >= segments['climb']*(1-percent_diff/100)) & (df['climb'] <= segments['climb']*(1+percent_diff/100)) &
                (df['seg_distance'] >= segments['seg_distance']*(1-percent_diff/100)) & (df['seg_distance'] <= segments['seg_distance']*(1+percent_diff/100))]

    # Group the data by trail name
    grouped_data = subset.groupby("name")

    # Now you can access the trails that have segments that fall within the given range
    return grouped_data

def euclidean_distance_zero_pad(ts):
    # Get the length of the time series
    length = len(ts)
    # Create a zero-vector of the same length as the time series
    zero_vector = np.zeros(ts.shape)
    # Pad the time series with the zero-vector
    padded_ts = np.concatenate((ts, zero_vector[len(ts):]))
    # Compute the Euclidean distance
    distance = np.sqrt(np.sum((padded_ts - zero_vector)**2))
    return distance

DB = pd.read_parquet('data/TrackDataBaseNB.parquet')
SDB = pd.DataFrame()
FDB = pd.DataFrame()
Columns=['curvature','climb','seg_distance']

# print(tabulate(DB.loc[DB['name']==1],headers='keys',tablefmt='github'))
print(DB.columns)
for i in range(0,DB['name'].values[-1]+1):
    TDF=DB.loc[DB['name']==i]
    segmented_DF=DF_to_segmented_DF(TDF)
    SDB = pd.concat([SDB,segmented_DF],axis=0)
    FDF=segments_to_feature_df(segmented_DF)
    classify_feature_df(FDF)
    cost = euclidean_distance_zero_pad(FDF[Columns].values)
    FDF['name']=i
    FDF['cost']=int(cost)
    FDB = pd.concat([FDB,FDF],axis=0)
    # print(tabulate(FDF))
    print(i,cost)
print(tabulate(FDB,headers='keys',tablefmt='github'))