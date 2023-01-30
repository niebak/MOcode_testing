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



def add_to_database(TDF0,databasename='data/TrackDataBaseNB.parquet',variables=['name','position_lat','position_long','altitude']):
    '''
    Add a dataframe to a database. Assume the database Ive created earlier.
    doesnt return anything, and doesnt add duplicates. creates a new DB on prompt if no database is found
    '''
    if 'os' not in globals():
        import os
    # Either read or create database and add the track to it:
    databaseexists = os.path.exists(databasename)
    if databaseexists:
        DBDF=pd.read_parquet(databasename)
        print('Found Database')
        LatestTrackName=DBDF['name'].iloc[DBDF.shape[0]-1]
        NewTrackNameVal=LatestTrackName+1
        FlagForDupes=0# zero if the track is already in the database
        # Check if the course is in any of the known courses
        addedname=[NewTrackNameVal]*TDF0.shape[0]
        TDF0['name']=addedname
        DFtoadd=TDF0[variables]
        for i in range(0,LatestTrackName+1):# Since it is non-inlcusive, we need to run to one more
            TrackCheck=DBDF.loc[DBDF['name']==i]
        #     print(tabulate(LatestTrack.iloc[0:10], headers = 'keys', tablefmt = 'github'))
            print(f'looking at track {i}')
            newlat=list(DFtoadd['position_lat'])
            oldlat=list(TrackCheck['position_lat'])
            newlong=list(DFtoadd['position_long'])
            oldlong=list(TrackCheck['position_long'])
            if(newlat==oldlat)and(newlong==oldlong):# Check if the coordinates are already present
                print(f'The provided track is already in the database as track {i}! please choose another, or remove the old one')
                FlagForDupes=1
        if not(FlagForDupes):# Runs if the dupe flag isnt set
            #i.e. runs if the track should be added to the database
            print('should print')
            DBDF=pd.concat([DBDF,DFtoadd])
            DBDF.to_parquet(databasename)
    else:
        print('Failed to find database.')
        ans=input('Should we create a new one?\n y/n\n')
        if ans=='y':
            print('Creating a new database')
            DBDF=pd.DataFrame()
            TDF1=DFtoadd
            trackname=[0]*TDF1.shape[0]
            TDF1['name']=trackname
            DBDF=TDF1[['name','position_lat','position_long','altitude']]
            databaseexists=True
            print(tabulate(DBDF.iloc[0:10], headers = 'keys', tablefmt = 'github'))
            DBDF.to_parquet(databasename)
        else:
            print('Wont create new database.\nPlease add database file to the path.')