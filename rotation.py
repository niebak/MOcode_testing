import pandas as pd
from code_functions import DF_to_segmented_DF,print_df,fit_records_to_frame,plot_segments_and_trail
import numpy as np
import matplotlib.pyplot as plt

def angle_with_x_axis(start, end):
    """
    Computes the angle (in radians) between the vector going from start to end and the x-axis.
    """
    if 'math' not in globals():
        import math
    # Compute the vector between the start and end points
    x = end[0] - start[0]
    y = end[1] - start[1]
    
    # Compute the angle between the vector and the x-axis
    angle = math.atan2(y, x)
    
    return angle
def dist(x, y, latline, lonline):
    if 'math' not in globals():
        import math
    dists = []
    for i in range(len(x)):
        closest_dist = float('inf')
        closest_lat, closest_lon = None, None
        for j in range(len(latline)-1):
            x1, y1 = latline[j], lonline[j]
            x2, y2 = latline[j+1], lonline[j+1]
            xp = x1 + (x2 - x1) * ((x[i] - x1) * (x2 - x1) + (y[i] - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
            yp = y1 + (y2 - y1) * ((x[i] - x1) * (x2 - x1) + (y[i] - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
            dist = math.sqrt((xp - x[i])**2 + (yp - y[i])**2)
            vec1 = [xp - x[i], yp - y[i], 0]
            vec2 = [x2 - x1, y2 - y1, 0]
            cross = np.cross(vec1, vec2)
            if cross[2] > 0:
                dist *= -1  # point is above line
            closest_dist = dist if dist < closest_dist else closest_dist
            closest_lat = xp if dist < closest_dist else closest_lat
            closest_lon = yp if dist < closest_dist else closest_lon
        dists.append(closest_dist)
    return dists


TDF0 = DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))
# TDF0 = DF_to_segmented_DF(fit_records_to_frame('data/Evening_Run.fit'))
# plot_segments_and_trail(TDF0,x_axis='Vposition_long',y_axis='Vposition_lat',Show_Plot=True)



for i in np.unique(TDF0['segments'].to_numpy()):
    wdf = TDF0.loc[TDF0['segments']==i].copy(deep=True)
    # wdf['position_lat']=wdf['Vposition_lat'].cumsum()
    # wdf['position_long']=wdf['Vposition_long'].cumsum()
    
    # plot_segments_and_trail(wdf,Show_Plot=True)
    
    x = wdf['position_lat'].to_numpy()
    y = wdf['position_long'].to_numpy()
    latline = np.linspace(x[0],x[-1],len(x))
    lonline = np.linspace(y[0],y[-1],len(y))
    
    ax0,ax1 = plot_segments_and_trail(wdf)
    ax1.plot(lonline,latline)
    distances = dist(x,y,latline,lonline)
    print(distances)
    plt.show()
    # Rotate the graph
    angle = angle_with_x_axis((latline[0],lonline[0]),(latline[-1],lonline[-1]))  # 45 degrees
    print(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = np.column_stack((x, y))
    rotated_xy = np.dot(xy, rotation_matrix)
    rotated_x, rotated_y = rotated_xy[:,0], rotated_xy[:,1]

    #Plot the original graph
    fig=plt.figure()
    fig2=plt.figure()
    ax2=fig.add_subplot(1,1,1)
    ax3=fig2.add_subplot(1,1,1)
    ax2.plot(x, y)
    ax2.grid()
    # ax2.plot(lonline,latline)
    ax2.set_title('Original Graph')
    #Plot the rotated graph
    # ax3.plot(rotated_x, rotated_y)
    ax3.stem(distances)
    ax3.grid()
    ax3.set_title('stemmed Graph')
    plt.show()