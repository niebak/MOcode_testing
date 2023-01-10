def coordinate_to_vector(point1, point2):
    '''
    Takes as input two three dimensional points and returns the distance between them in the dimensions.
    Return is a vector of the changes
    '''
    x=point2[0]-point1[0]
    y=point2[1]-point1[1]
    z=point2[2]-point1[2]
    return (x, y, z)
def cumulative_sum_with_limit(list1, list2, list3,l1=0.003,l2=0.003,l3=5,new_segment_marker=-1):
    '''
    running through three lists and outputting a marker list if a limit is reached.
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
    segmentlist=Marker_List
    segmentname=initial_segment
    for i in range(0,len(Marker_List)):
        curr=Marker_List[i]
        if(curr==segment_marker):
            segmentname=segmentname+1
        segmentlist[i]=segmentname
    return segmentlist
