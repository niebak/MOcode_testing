if 'pandas' not in globals():
    import pandas as pd
if 'matplotlib.pyplot' not in globals():
    import matplotlib.pyplot as plt

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
def cumulative_sum_with_limit(list1, list2, list3,l1=0.009,l2=0.009,l3=5,new_segment_marker=-1):
    '''
    Running through three lists and outputting a marker list if a limit is reached.
    '''
    cum_sum1=0
    cum_sum2=0
    cum_sum3=0
    markers = [-2]*len(list1)
    for i in range(len(list1)):
        cum_sum1 += list1[i]
        if cum_sum1 > l1:
            markers[i]=new_segment_marker
            cum_sum1=0
            cum_sum2=0
            cum_sum3=0
        cum_sum2 += list2[i]
        if cum_sum2 > l2:
            markers[i]=new_segment_marker
            cum_sum1=0
            cum_sum2=0
            cum_sum3=0
        cum_sum3 += list3[i]
        if cum_sum3 > l3:
            markers[i]=new_segment_marker
            cum_sum1=0
            cum_sum2=0
            cum_sum3=0
    return markers
def marker_to_segment(Marker_List,segment_marker=-1,initial_segment=-1):
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
                            segment_column='segments',Show_plot=False):
    '''
    Returns two axessubplots. Can also plot the subplots directly with Show_plot.
    Takes a dataframe, and can also be given what columns to plot.
    '''
    fig = plt.figure()
    ax0=fig.add_subplot(2,1,1)
    segments=TDF0[segment_column].tolist()
    ax0.plot(TDF0[x_axis],TDF0[y_axis])
    ax0.grid()
    ax1=fig.add_subplot(2,1,2)
    for i in range(segments[0],segments[-1]+2):
        testdf=TDF0.loc[TDF0[segment_column]==i]
        ax1.plot(testdf[x_axis],testdf[y_axis])
    ax1.grid()
    if Show_plot:
        plt.show()
    return ax0,ax1