from code_functions import *

# The functinos are present in the bottom, but also in code_functions
# An example course
coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data

TDF0=DF_to_segmented_DF(pd.read_excel(coursename),Weird_format=True)
print(TDF0)
FDF0=segments_to_feature_df(TDF0)
print(FDF0)
classify_feature_df(FDF0)
print(FDF0)

def segments_to_feature_df(TDF0):
    '''
    Goes from a segmented df to a feature df, which is only segments and their features
    '''
    curvature= [0]*(TDF0['segments'].iloc[-1]+1)
    climb = [0]*(TDF0['segments'].iloc[-1]+1)
    seg_dist = [0]*(TDF0['segments'].iloc[-1]+1)

    for i in range(TDF0['segments'].iloc[-1]+1):
        segment = TDF0.loc[TDF0["segments"]==i]
        curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,1)
        if abs(curvature[i])<1:
            curvature[i]=0
        climb[i]=calculate_height_gained(segment)
        if abs(climb[i])<1:
            climb[i]=0
        seg_dist[i]=find_distance(segment)
    featdict={'curvature':curvature,'climb':climb,'seg_distance':seg_dist}
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