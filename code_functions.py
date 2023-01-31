if 'pandas' not in globals():
    import pandas as pd
if 'matplotlib.pyplot' not in globals():
    import matplotlib.pyplot as plt
if 'numpy' not in globals():
    import numpy as np
from fitparse import FitFile
import os
# To limit memory leak, will result in a higher runtime
os.environ["OMP_NUM_THREADS"] = "1"

def coordinate_to_vector(point1, point2):
    '''
    Takes as input two three dimensional points and returns the distance between them in the dimensions.
    Return is a vector of the changes.
    '''
    # Of the form delta x, delta y, delta z between the two points.
    x=point2[0]-point1[0]
    y=point2[1]-point1[1]
    z=point2[2]-point1[2]
    return (x, y, z)
def coordinate_to_vector_dataframe(TDF0,column1="position_lat",column2="position_long",column3="altitude"):
    '''
    Takes as input a DF with coordinates and expresses the coordinates as a series of changes from the previous point.
    Returns a new DF with complete coordinates.
    '''
    coordinates=TDF0[[column1,column2,column3]]
    VectorCoordinates=[coordinates.iloc[0].tolist()]*coordinates.shape[0]
    for point in range(0,coordinates.shape[0]-1):
        VectorCoordinates[point+1]=coordinate_to_vector(coordinates.iloc[point].tolist(),coordinates.iloc[point+1].tolist())
    VectorDataframe=pd.DataFrame(VectorCoordinates,columns=['Vposition_lat',"Vposition_long","Valtitude"])
    return VectorDataframe
def cumulative_sum_with_limit(list1, list2, list3, l1=1, l2=1, l3=5, new_segment_marker=-1, min_consec=10):
    cum_sum1 = 0
    cum_sum2 = 0
    cum_sum3 = 0
    markers = [-2] * len(list1)
    consec_count = 0
    for i in range(len(list1)):
        cum_sum1 += list1[i]
        if cum_sum1 > l1:
            consec_count += 1
        
        if consec_count >= min_consec:
            markers[i] = new_segment_marker
            consec_count = 0
            cum_sum1=0
            cum_sum2=0
            cum_sum3=0

        cum_sum2 += list2[i]
        if cum_sum2 > l2:
            consec_count += 1
        
        if consec_count >= min_consec:
            markers[i] = new_segment_marker
            consec_count = 0
            cum_sum1=0
            cum_sum2=0
            cum_sum3=0

        cum_sum3 += list3[i]
        if cum_sum3 > l3:
            consec_count += 1
        
        if consec_count >= min_consec:
            markers[i] = new_segment_marker
            consec_count = 0
            cum_sum1=0
            cum_sum2=0
            cum_sum3=0
    return markers
def marker_to_segment(Marker_List,segment_marker=-1,initial_segment=0):
    '''
    Takes a marker-list and creates a segmentlist.
    '''
    segmentlist=Marker_List
    segmentname=initial_segment
    for i in range(0,len(Marker_List)):
        curr=Marker_List[i]
        if(curr==segment_marker):
            segmentname=segmentname+1
        segmentlist[i]=segmentname
    return segmentlist
def detect_and_mark_change_in_direction(list1, list2, list3, threshold=10, change_limit=0.01):
    '''
    Takes three lists and returns a marker list when consecutive values in the lists change direction.
    Has a insensitivity limit. 
    '''
    growth_direction = None
    marker_list = [0]*len(list1)
    change_count = 0
    for i in range(1, len(list1)):
        if growth_direction is None:
            if (list1[i] - list1[i - 1]) > change_limit or (list2[i] - list2[i - 1]) > change_limit or (list3[i] - list3[i - 1]) > change_limit:
                growth_direction = "increasing"
            elif (list1[i] - list1[i - 1]) < -change_limit or (list2[i] - list2[i - 1]) < -change_limit or (list3[i] - list3[i - 1]) < -change_limit:
                growth_direction = "decreasing"
        elif growth_direction == "increasing" and ((list1[i] - list1[i - 1]) < -change_limit or (list2[i] - list2[i - 1]) < -change_limit or (list3[i] - list3[i - 1]) < -change_limit):
            change_count += 1
            if change_count >= threshold:
                marker_list[i] = -1
                change_count = 0
                growth_direction = "decreasing"
        elif growth_direction == "decreasing" and ((list1[i] - list1[i - 1]) > change_limit or (list2[i] - list2[i - 1]) > change_limit or (list3[i] - list3[i - 1]) > change_limit):
            change_count += 1
            if change_count >= threshold:
                marker_list[i] = -1
                change_count = 0
                growth_direction = "increasing"
    return marker_list
def plot_segments_and_trail(TDF0,x_axis='position_long',y_axis='position_lat',
                            segment_column='segments',Show_Plot=False):
    '''
    Returns two axessubplots. Can also plot the subplots directly with Show_Plot.
    Takes a dataframe, and can also be given what columns to plot.
    '''
    fig = plt.figure()
    ax0=fig.add_subplot(2,1,1)
    segments=TDF0[segment_column].tolist()
    ax0.plot(TDF0[x_axis],TDF0[y_axis],label='Complete trail')
    ax0.grid()
    ax1=fig.add_subplot(2,1,2)
    for i in range(segments[0],segments[-1]+1):
        testdf=TDF0.loc[TDF0[segment_column]==i]
        ax1.plot(testdf[x_axis],testdf[y_axis],label=i)
    ax1.grid()
    if Show_Plot:
        plt.show()
    return ax0,ax1
def fit_records_to_frame(fitfile, vars=[], max_samp=36000):
    '''Returnerer en dataframe med valgfrie variabler per records fra fitfilen.
    Variabler man ønsker må legges inn i vars, f.eks.:
    vars = ['position_lat', 'position_long', 'altitude', 'distance',
    'enhanced_altitude']
    dataframe = fit_records_to_frame(fitfile, vars)
    Som default vil timestamp, heart_rate og power returneres.
    max_samp angir begrensning i hvor mange samples (sekunder) man maksimalt
    kan hente ut. Standard tilsvarer 10 timer, som burde holde for de
    fleste .fit-filer.
    '''
    if 'timestamp' in vars:
        vars.remove('timestamp')
    time = np.empty(max_samp, dtype='datetime64[s]')
    data = np.empty((max_samp, len(vars)))
    fit = FitFile(fitfile)
    for i, rec in enumerate(fit.get_messages('record')):
        for rec_data in rec:
            if rec_data.name in vars and rec_data.value != None:
                data[i, vars.index(rec_data.name)] = rec_data.value
            elif rec_data.name == 'timestamp':
                time[i] = rec_data.value
    frame = pd.DataFrame(data[:i+1, :], columns=vars)
    droplist = [i for i in frame.columns if all(v!=v for v in frame[i])]
    frame.drop(droplist, axis=1, inplace=True)
    frame = frame.assign(timestamp=pd.Series(time[:i+1]).values)
    return frame
def DF_to_segmented_DF(DF,threshold=10):
    '''
    Assumes that the dataframe has position_lat, position_long, and altitude.
    Uses detect and mark change
    '''
    TDF0 = DF.copy()
    limits=[180,90] # lat and lon are within these limits(+-)
    if max(TDF0['position_lat'].values)>limits[0]:
        # If the data comes from strava/garmin it is in a weird format
        TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
        TDF0['position_long'] = TDF0['position_long']/(2**32/360)
        print(f'\nAltered the trail so that it is in lat/long __POSSIBLY__\n')  
    if 'distance' not in TDF0.columns:
        TDF0=cum_haversine_distance(TDF0)
    # Create the vector-representation coordinates
    TDF0_vector = coordinate_to_vector_dataframe(TDF0)
    TDF0_seg_marker=detect_and_mark_change_in_direction(
        TDF0_vector['Vposition_lat'].tolist(),
        TDF0_vector['Vposition_long'].tolist(),
        TDF0_vector['Valtitude'].tolist(),
        threshold=threshold)
    TDF0_seg =  marker_to_segment(TDF0_seg_marker, initial_segment=0)
    TDF0=pd.concat([TDF0,TDF0_vector],axis=1)
    TDF0['segments']=TDF0_seg
    if 'timestamp' in TDF0.columns:
        add_velocity_column(TDF0)
        TDF0['seconds'] = TDF0['timestamp'].diff()
    else:
        print('Have not added velocity column to Dataframe, errors might come as a result of this')
    if 'timestamp' not in TDF0.columns and 'speed' in TDF0.columns:
        from datetime import datetime,timedelta
        TDF0['seconds'] = (TDF0['distance'].diff()/TDF0['speed']).cumsum()
        TDF0['seconds'].iloc[0]=0
        TDF0['timestamp'] = TDF0.apply(lambda row: datetime.utcfromtimestamp(row['seconds']), axis=1)
        print('Added timestamp, will lack date and year')
    if 'speed' in TDF0.columns:
        TDF0.rename(columns={'speed':'velocity [m/s]'},inplace=True)
        print('Renamed "speed" to "velocity [m/s]"')
    
     # Check only the last segment
    last_segment = TDF0.loc[TDF0["segments"]== TDF0['segments'].iloc[-1]]
    if len(last_segment) < threshold:
        # Merge the last segment with the previous segment
        TDF0.loc[TDF0["segments"] == TDF0['segments'].iloc[-1], "segments"] = TDF0['segments'].iloc[-1]-1
    return TDF0
def create_segmentDF_fromDF(TDF2,variables='all'):
    '''
    Goes from a long dataframe to a dataframe defined as the mean of each segment representing each segment.
    Uses variables=['Vposition_lat','Vposition_long','Valtitude','segments'], and returns a dataframe
    '''
    
    if variables=='all':
        SDF2 = TDF2.groupby('segments').mean()
    else:
        SDF2 = TDF2[variables].groupby('segments').mean()
    SDF2['segments']=np.unique(TDF2['segments'].tolist())
    return SDF2
def find_distance(df):
    '''
    Works on segments. If distance is not present, it will calculate the distance using euc. distance.
    Returns distance from start to end.
    '''
    if 'distance' not in df.columns:
        df=cum_haversine_distance(df)
    start=df['distance'].iloc[0]
    stop=df['distance'].iloc[-1]
    return round(stop-start,2)
def calculate_curvature(dataframe):
    '''
    Works on segments. Assumes I have position_long and position_lat available. 
    Returns curvature over the __entire__ dataframe.
     Change with Calsulate_distance_from_straight_line
    '''
    x = dataframe['position_long'].values
    y = dataframe['position_lat'].values
    dx = np.gradient(x)
    dy = np.gradient(y)
    curvature = np.zeros(len(dx))
    for i in range(len(dx)):
        if dx[i] == 0 or dy[i] == 0:
            curvature[i] = 0
        else:
            curvature[i] = (dy[i] / dx[i]) / (1 + (dy[i] / dx[i])**2)**(3/2)
    return np.sum(curvature)
def calculate_distance_from_straight_line(df):
    '''
    Can be used to quantify the degree of a turn by returning the distance it is from a straight line
    from its start to end point. Works on segments.
    Returns the mean.
    '''
    x1, y1 = df.iloc[0]['position_long'], df.iloc[0]['position_lat']
    x2, y2 = df.iloc[-1]['position_long'], df.iloc[-1]['position_lat']
    if (x2-x1) != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        m = 0
    c = y1 - (m * x1)
    distance = []
    for i in range(len(df)):
        x,y = df.iloc[i]['position_long'], df.iloc[i]['position_lat']
        distance.append((y - (m * x + c)))
    return np.mean(distance)
def calculate_height_gained(dataframe):
    '''
    Assumes altitude is present in dataframe. Works on segments
    Returns altitude difference in begining and end.
    '''
    start = dataframe['altitude'].iloc[0]
    stop = dataframe['altitude'].iloc[-1]
    height_gained = stop - start
    return round(height_gained,4)
def cum_haversine_distance(df,R = 6371*10**3):
    '''
    Adds the haversine distance in a seperate column.
    Assumes earths radius to be 6371km, but this can be changed.
    '''
    if 'math' not in globals():
        import math
    df["pp_distance"] = 0.0
    for i in range(len(df)-1):
        lat1, lon1, lat2, lon2 = map(math.radians, [df.at[i, "position_lat"], df.at[i, "position_long"], df.at[i+1, "position_lat"], df.at[i+1, "position_long"]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c
        df.at[i, "pp_distance"] = d
    df["distance"] = df["pp_distance"].cumsum()
    return df
def add_velocity_column(df,inplace=True,add_time=False):
    '''
    Takes as input a dataframe with columns 'timestamp' and 'distance' and returns the velocity
    in m/s and km/h.
    '''
    velocity = df['distance'].diff() / (df['timestamp'].diff().dt.total_seconds())
    velocity[0]=0
    if add_time:
        df['total_seconds'] = (df['timestamp'].diff().dt.total_seconds().cumsum())
    if inplace:
        df['velocity [m/s]'] = velocity
        df['velocity [km/h]'] = df['velocity [m/s]'] * 3.6
    if not(inplace):
        return velocity
def segments_to_feature_df(TDF0):
    '''
    Goes from a segmented df to a feature df, which is only segments and their features.
    Curvature, Climb, and Distance travelled in the segment.
    '''
    curvature= [0]*(TDF0['segments'].iloc[-1]+1)
    climb = [0]*(TDF0['segments'].iloc[-1]+1)
    seg_dist = [0]*(TDF0['segments'].iloc[-1]+1)
    segments = TDF0['segments'].tolist()
    for i in range(TDF0['segments'].iloc[-1]+1): # Loop through each segment
        segment = TDF0.loc[TDF0["segments"]==i]
        if segment.shape[0]==0:
            print(f'problem at index {i}, do some troubleshooting.')
        else:
            # find the curvature as a distance from a straight line
            curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,1)  
            if abs(curvature[i])<1:
                curvature[i]=0
            # Find the altitude difference
            climb[i]=calculate_height_gained(segment)
            if abs(climb[i])<1:
                climb[i]=0
            # find the distance
            seg_dist[i]=find_distance(segment)
    featdict={'segments':np.unique(segments),'curvature':curvature,'climb':climb,'seg_distance':seg_dist}
    featureDF=pd.DataFrame(featdict)
    return featureDF
def classify_feature_df(featureDF,curve_lim=1,climb_lim=3,inplace=True):
    '''
    Classifies a featureDF according to some criteria on curves/climbs. Has the following space:
    [R/L turn or straight] [incline/decline/empty]
    so [3x3] classes,+ 1 "unknown class" if none is applicable.
    either returns a list of classes, or adds a column. 
    '''
    # Trying to write a classifier
    Segment_Class=['']*featureDF.shape[0]
    for segment in range(0,featureDF.shape[0]):
        curve=featureDF['curvature'].iloc[segment]
        climb=featureDF['climb'].iloc[segment]
        distance=featureDF['seg_distance'].iloc[segment]
        if curve> curve_lim:
            Segment_Class[segment] = 'R turn'
        if abs(curve)<=curve_lim:
            Segment_Class[segment]='straight'
        if curve<-curve_lim:
            Segment_Class[segment]='L turn'
        if climb>=climb_lim:
            Segment_Class[segment]+=(' incline')
        if climb<=-climb_lim:
            Segment_Class[segment]+=(' decline')
        if Segment_Class[segment] == '':
            Segment_Class[segment] = 'Unknown class'
    if inplace:
        featureDF['class']=Segment_Class
    else:
        return Segment_Class
def df_curve_to_signal_rep(df):
    '''
    Might be used in frequency analysis, but we'll see. returns a vector of magnitudes.
    '''
    if 'datetime' not in globals():
        import datetime
    x1, y1 = df.iloc[0]['position_long'], df.iloc[0]['position_lat']
    x2, y2 = df.iloc[-1]['position_long'], df.iloc[-1]['position_lat']
    m = (y2 - y1) / (x2 - x1)
    c = y1 - (m * x1)
    distance = []
    for i in range(len(df)):
        x,y = df.iloc[i]['position_long'], df.iloc[i]['position_lat']
        distance.append((y - (m * x + c)))
    time = (df['timestamp'].diff().tolist())
    if any(isinstance(i, datetime.timedelta) for i in time):
        time = (df['timestamp'].diff().dt.total_seconds().cumsum().tolist())
    return distance,np.nan_to_num(time)
def frequency_analysis(TDF0):
    '''
    Does some Fourier analysis and returns the maximum value and its frequency.
    Works on RAW TRAILS, but returns SEGMENT DATAFRAME.
    '''
    if 'timestamp' not in TDF0.columns:
        pp_time = [0]*TDF0.shape[0]
        pp_distance=TDF0['distance'].diff()
        for i in range(0,TDF0.shape[0]):
            pp_time[i] = pp_distance[i]/TDF0['velocity [m/s]'].iloc[i]
        pp_time[0]=0
        timestamp = np.cumsum(pp_time)
        TDF0['pp_time'] = pp_time
        TDF0['timestamp']=timestamp
    SDF0=(segments_to_feature_df(TDF0))
    classify_feature_df(SDF0) 
    segments = TDF0['segments'].tolist()
    amps=[0]*(segments[-1]+1)
    freqs=[0]*(segments[-1]+1)
    for i in range(segments[-1]+1):
        segment = TDF0.loc[TDF0['segments']==i]
        signal,time = df_curve_to_signal_rep(segment)
        fourier = np.delete(fourier_transform(signal),0)
        frequencies = np.delete(extract_frequencies_time(signal, time),0)
        amps[i]= max(fourier)
        freqs[i] = frequencies[np.argmax(fourier)]

    SDF0['freq_amplitude']=amps
    SDF0['freq']=freqs
    return SDF0
def extract_frequencies_time(magnitudes, time):
    from scipy.fftpack import fft,fftfreq
    fourier = fourier_transform(magnitudes)
    # Number of samples in the original signal
    N = len(magnitudes)
    # Time duration of the signal
    T = np.cumsum(time)[len(time)-1]
    # Sample rate
    sample_rate = N / T
    # Extract the frequencies
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)
    return frequencies
def fourier_transform(magnitudes):
    from scipy.fftpack import fft
    # Convert the input list to a numpy array
    data = np.array(magnitudes)

    # Perform Fourier Transform on the data
    fourier = np.abs(fft(data))

    return fourier
def add_to_database(TDF2,databasename='data/TrackDataBaseNB.parquet',variables=['timestamp','position_lat','position_long','altitude','segments','velocity [m/s]']):
    '''
    Add a dataframe to a database. Assume the database Ive created earlier.
    doesnt return anything, and doesnt add duplicates. creates a new DB on prompt if no database is found
    '''
    try:
         tabulate
    except NameError:
        import tabulate
    try:
        os
    except NameError:
        import os
    # Either read or create database and add the track to it:
    databaseexists = os.path.exists(databasename)
    TDF0 = TDF2[variables]
    if databaseexists:
        if '.xlsx' in databasename:
            DBDF = pd.read_excel(databasename)
            if 'Unnamed: 0' in DBDF.columns:
                del DBDF['Unnamed: 0']
        if '.parquet' in databasename:
            DBDF = pd.read_parquet(databasename)
            if 'Unnamed: 0' in DBDF.columns:
                del DBDF['Unnamed: 0']
        print('Found Database')
        LatestTrackName=DBDF['name'].iloc[DBDF.shape[0]-1]
        NewTrackNameVal=LatestTrackName+1
        FlagForDupes=False# zero if the track is already in the database
        # Check if the course is in any of the known courses
        addedname=[NewTrackNameVal]*TDF0.shape[0]
        TDF0['name']=addedname
        DFtoadd=TDF0
        for i in range(0,LatestTrackName+1):# Since it is non-inlcusive, we need to run to one more
            TrackCheck=DBDF.loc[DBDF['name']==i]
        #     print(tabulate(LatestTrack.iloc[0:10], headers = 'keys', tablefmt = 'github'))
            print(f'looking at track {i}')
            # if (DFtoadd.shape[0]==TrackCheck.shape[0]) and ('feature' not in databasename) :
            #     mask = np.isclose(DFtoadd['position_lat'],TrackCheck['position_lat'],rtol=0.001)
            #     FlagForDupes=mask.all()
            #     print(f'The track is already present, as track {i}.\n')
            #     break
        if not(FlagForDupes):# Runs if the dupe flag isnt set
            #i.e. runs if the track should be added to the database
            # print(DBDF.columns)
            # print(DFtoadd.columns)
            DBDF=pd.concat([DBDF,DFtoadd],ignore_index=True)
        if '.xlsx' in databasename:
            DBDF.to_excel(databasename,index = False)
        if '.parquet' in databasename:
            DBDF.to_parquet(databasename)
    else:
        print('Failed to find database.')
        ans=input('Should we create a new one?\n y/n\n')
        DFtoadd=TDF0[variables]
        if ans=='y':
            print('Creating a new database')
            DBDF=pd.DataFrame()
            TDF1=DFtoadd
            trackname=[0]*TDF1.shape[0]
            TDF1['name']=trackname
            variables.append('name')
            DBDF=TDF1[variables]
            databaseexists=True
            print(tabulate.tabulate(DBDF.iloc[0:10], headers = 'keys', tablefmt = 'github'))
            if '.xlsx' in databasename:
                DBDF.to_excel(databasename,index = False)
            if '.parquet' in databasename:
                DBDF.to_parquet(databasename)
        else:
            print('Wont create new database.\nPlease add database file to the path.')
def find_matching_trails(df, segments, percent_diff):
    '''
    This works as a prototype filter, although it has to be expanded upon ion order to work on the database.
    Steps to be done:
    1. expand use to database insead of dataframe
    1.1 can be done by having the functino return true/false and apply to each track
    2. create a system that returns a set amount of trail, i.e. the top X% where the X is moved automatically.
    '''
    # Create a subset of the dataframe that contains the segments within the range and percentage difference
    subset = df[(df['class'] == segments['class']) & 
                (df['curvature'] >= segments['curvature']*(1-percent_diff/100)) & (df['curvature'] <= segments['curvature']*(1+percent_diff/100)) &
                (df['climb'] >= segments['climb']*(1-percent_diff/100)) & (df['climb'] <= segments['climb']*(1+percent_diff/100)) &
                (df['seg_distance'] >= segments['seg_distance']*(1-percent_diff/100)) & (df['seg_distance'] <= segments['seg_distance']*(1+percent_diff/100))]

    # Group the data by trail name
    grouped_data = subset.groupby("name")

    # Now you can access the trails that have segments that fall within the given range
    return grouped_data
def segments_to_feature_df_with_rev(TDF0):
    '''
    Goes from a segmented df to a feature df, which is only segments and their features.
    Curvature, Climb, and Distance travelled in the segment.
    '''
    segments = np.unique(TDF0['segments'].tolist())
    curvature= [0]*(len(segments))
    climb = [0]*(len(segments))
    seg_dist = [0]*(len(segments))
    for i in range(len(np.unique(segments))): # Loop through each segment
        seg = segments[i]
        segment = TDF0.loc[TDF0["segments"]==seg]
        # find the curvature as a distance from a straight line
        curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,1)  
        if abs(curvature[i])<0.01:
            curvature[i]=0
        # Find the altitude difference
        climb[i]=calculate_height_gained(segment)
        if abs(climb[i])<1:
            climb[i]=0
        # find the distance
        seg_dist[i]=find_distance(segment)
    featdict={'segments':np.unique(segments),'curvature':curvature,'revcurvature':[i * -1 for i in curvature],'climb':climb,'revclimb':[i * -1 for i in climb],'seg_distance':seg_dist}
    featureDF=pd.DataFrame(featdict)
    return featureDF