#Wave Circle Percentile Fitting Script: 
#Strickland, Evelyn. 2022, v1.0
#Bundled with 2022-analysis-pipeline

#INPUT: ARCOS Dataframe from run_arcos_set.py script process. 
#Intended for use in the analysis set framework software!

#OUTPUT: A dataframe with the waves fit by the minimum circle fit to any points in the ARCOS cloud at a given time point that are also within
#some percentile of the mean of the cloud. 
#Intended to balance automation with human in the loop elimination of spurious circle fitting and wave detection. 
#For example, users will be able to eliminate fits to events not coming from the center core swarm, or event splitting that 
#is accidentally coming out of the ARCOS algorithm (happens in larger waves)

#Importing software, napari, plotting, numpy, and paths math etc...
import napari
from napari import Viewer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import math, random

#Image Tooling: 
from skimage.io import imread

#Convienient hex colormap
TAB20 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",]

#------------------------------------------------------------------------------------------------------------------------
#Min circle fitting method:
# Smallest enclosing circle - Library (Python)
# 
# Copyright (c) 2020 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle
# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).
# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
# 
# Initially: No boundary points known

def make_circle(points):
	# Convert to float and randomize order
	shuffled = [(float(x), float(y)) for (x, y) in points]
	random.shuffle(shuffled)
	
	# Progressively add points to circle or recompute circle
	c = None
	for (i, p) in enumerate(shuffled):
		if c is None or not is_in_circle(c, p):
			c = _make_circle_one_point(shuffled[ : i + 1], p)
	return c

# One boundary point known
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c

# Two boundary points known
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left  = None
	right = None
	px, py = p
	qx, qy = q
	
	# For each point not in the two-point circle
	for r in points:
		if is_in_circle(circ, r):
			continue
		
		# Form a circumcircle and classify it on left or right side
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
			right = c
	
	# Select which circle to return
	if left is None and right is None:
		return circ
	elif left is None:
		return right
	elif right is None:
		return left
	else:
		return left if (left[2] <= right[2]) else right

def make_diameter(a, b):
	cx = (a[0] + b[0]) / 2
	cy = (a[1] + b[1]) / 2
	r0 = math.hypot(cx - a[0], cy - a[1])
	r1 = math.hypot(cx - b[0], cy - b[1])
	return (cx, cy, max(r0, r1))

def make_circumcircle(a, b, c):
	# Mathematical algorithm from Wikipedia: Circumscribed circle
	ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
	oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
	ax = a[0] - ox;  ay = a[1] - oy
	bx = b[0] - ox;  by = b[1] - oy
	cx = c[0] - ox;  cy = c[1] - oy
	d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
	if d == 0.0:
		return None
	x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
	y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
	ra = math.hypot(x - a[0], y - a[1])
	rb = math.hypot(x - b[0], y - b[1])
	rc = math.hypot(x - c[0], y - c[1])
	return (x, y, max(ra, rb, rc))

_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
	return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
	return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

#------------------------------------------------------------------------------------------------------------------------
#Cloud --> Percentile Circle Fitting Code:
#passing the points[:,0] as x, points[:,1] as y, (a,b) as center
#from: https://stackoverflow.com/questions/60219540/convert-cartesian-coordinate-system-to-polar-coordinate-system-with-multi-points

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def percent_circle_fit(xs,ys, percentage):
    x_trans = xs - np.mean(xs)
    y_trans = ys - np.mean(ys)
    centroid = (np.mean(xs), np.mean(ys))
    points = np.array([[xi,yi] for xi,yi in zip(x_trans,y_trans)])
    radial_points = np.array([cart2pol(xi,yi) for xi,yi in zip(x_trans,y_trans)])

    radial_thresh = np.percentile(radial_points[:,0],percentage)
    inner_pts = []

    for i in range(len(radial_points)):
        if radial_points[i][0] <= radial_thresh:
            inner_pts.append(radial_points[i])
    inner_pts = np.array(inner_pts)

    points_retreive = np.array([pol2cart(point[0], point[1]) for point in inner_pts])
    
    x,y,r = make_circle(points_retreive)
    
    return (x+centroid[0],y+centroid[1]), r

def calculate_wave_circle_fit(image_data, df_arcos, time_step, px_to_um, image_path, wave_analysis_save_path, percentage = 95, image_contrast_min = 100, image_contrast_max = 1500):
    cl_group = df_arcos.groupby('clTrackID')
    wave_fit_collect = []
    
    save_dir_full = wave_analysis_save_path / ('circle_fit_' + image_path.stem)
    print(wave_analysis_save_path)
    try:
        save_dir_full.mkdir()
    except FileExistsError: 
        'Output path already exists, proceeding anyway.'
    
    for k, wave in cl_group:
        print('Fitting wave ' + str(k))

        wave_fit_df = pd.DataFrame()
    
        timepts = wave.timepoint.unique()

        #Get the density measured at each timepoint in the arcos event: 
        densitys = []
        for t in timepts:
            densitys.append(wave[wave['timepoint'] == t].cell_density.values[0])

        wave_fit_df['clTrackID'] = [k for dum in range(len(timepts))]
        
        if len(timepts)<5:
            continue
        
        r_coll = []
        center_fit_coll = []

        #IMPROVEMENTNOTE: 
        #This section here could be used in a multithreading context, this would speed up frame circle fitting.

        for i in timepts:
            x = wave.loc[wave['timepoint']<=i].x.values
            y = wave.loc[wave['timepoint']<=i].y.values
            
            
            center, r = percent_circle_fit(x,y,percentage)
            circle1 = plt.Circle(center, r, color='r', fill=False)

            #circle1 = plt.Circle(centroid, r_fit, color='r', fill=False)
            center_fit_coll.append(center)
            r_coll.append(r*px_to_um)

            '''
            figure, axes = plt.subplots(dpi=150)
            #plt.scatter(x,y, c='orange', s=1)
            plt.imshow(image_data[i], vmin=image_contrast_min, vmax=image_contrast_max)
            axes.add_patch(circle1)
            axes.set_aspect(1)
            plt.title('Wave #:' + str(k))
            figure_save_path = save_dir_full / ('wave' + str(k) + '_time-' + str(i)+'.png')
            plt.savefig(figure_save_path)
            plt.close()
            '''
            
        
        r_coll = np.array(r_coll)
        r_sq = (r_coll - r_coll[0])**2
        
        #r_coll_sq = r_coll**2
        v_s = np.gradient(r_coll)/time_step
        a_s = np.gradient(v_s)/time_step 
        wave_fit_df['timepoint'] = timepts
        wave_fit_df['rel_time'] = timepts - timepts[0]
        wave_fit_df['rel_r'] = r_coll - r_coll[0]
        wave_fit_df['r_squared'] = r_sq
        wave_fit_df['circle_radius'] = r_coll
        wave_fit_df['radius_velocity'] = v_s
        wave_fit_df['radius_acceleration'] = a_s
        wave_fit_df['wave_centroid'] = center_fit_coll
        wave_fit_df['cell_density'] = densitys
        wave_fit_df['trackable'] = True
        

        #Calculate wave split location by looking for peak in calcium signal: 
        avg_ca = []
        r_max = np.max(r_coll)

        #Get the average calcium pixel flour value over time for a bounded max area of the given wave:
        for t in timepts:
            mask = create_circular_mask(image_data.shape[1],image_data.shape[1],center=center_fit_coll[-1], radius=r_max/2.2)
            masked_img = image_data[t].copy()
            masked_img[~mask] = 0
            avg_ca.append(np.sum(masked_img)/np.sum(mask))

        #Find the max of the calcium signal, create a tracking array to split signals later based on max ca sig:
        max_ca = np.where(avg_ca == np.amax(avg_ca))[0][0]
        sig_cut = np.zeros(len(avg_ca))
        sig_cut[max_ca+1:] = 1

        #Append both calcium signal and pre-post max split array: 
        wave_fit_df['ca_avg_sig'] = avg_ca
        wave_fit_df['split'] = sig_cut  

        wave_fit_collect.append(wave_fit_df)


    #Build the dataframe and return it:
    return pd.concat(wave_fit_collect, ignore_index=True)

#------------------------------------------------------------------------------------------------------------------------
#Custom Napari Viewer, DF reading functions, and a tool to help draw a min bbox for a circle: 

def read_tuple_list(x):
    x_strip = x.strip('()').strip('[]')
    if len(x_strip.split(', ')) == 1:
        return np.array(x_strip.split(), dtype=float)
    else:
        return np.array(x_strip.split(', '), dtype=float)

def gen_circle_corners(center,r, conversion):
    r_scaled = r/conversion
    return [[center[1] - r_scaled, center[0]+r_scaled],[center[1] + r_scaled, center[0]+r_scaled], [center[1] + r_scaled, center[0]-r_scaled], [center[1] - r_scaled, center[0]-r_scaled]]

def cleanup_wave_tracks_viewer(image_data, wave_df, colormap):
    clTracks = wave_df.groupby('clTrackID')
    clIDs = wave_df['clTrackID'].unique()
    colors = np.random.choice(colormap, size=len(clIDs))
    viewer = Viewer()
    # create one random polygon per "plane"
    contrast_list = [np.min(image_data), np.max(image_data)*0.1]
    viewer.add_image(image_data, name='Calcium image', contrast_limits=contrast_list, colormap='inferno')
    for i,(event_id, cl_event) in enumerate(clTracks): 
        timepts = cl_event.timepoint.values
        centroids = cl_event.wave_centroid.values
        radii = cl_event.circle_radius.values
        planes = np.tile(timepts.reshape((len(timepts),1,1)), (1,4,1))
        corners = np.array([gen_circle_corners(c,r,px_to_um_convert) for c,r in zip(centroids,radii)])
        shapes = np.concatenate((planes, corners), axis=2)
        viewer.add_shapes(
            np.array(shapes),
            shape_type='ellipse',
            face_color = '#ffffff00',
            edge_color=colors[i],
            name=str(event_id),
            opacity=1,
            visible=cl_event.trackable.values[0]
            )
    return viewer

#From: https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

#------------------------------------------------------------------------------------------------------------------------
#Run code below here: 
#Note, this will mainly be just pointed at analysis set folders, then do its thing in fitting the ARCOS DFs found in each file in the set.
#At each point, a napari viewer will display how the analysis is doing after circle fitting, and the user can cancel an analysis or retry with a different percentile. 

#Set this to the base path for an analysis set:
#Now, define a place to import to:
i = 0
while i < 1:
    #Take user input
    #Do you want to run a full analysis or expedited? (f/e):
    mode_query = input("Do you want to run a full or expedited analysis? (f/e): ")
    if mode_query == 'e':
        analysis_mode = 0
        print('Mode set to expidited.')
    else:
        analysis_mode = 1
        print('Mode set to full.')

    #Get the path directory
    analysis_path_potential = input("Enter the set to analyze: ")
    base_path = Path(analysis_path_potential)

    #Check if this path is a valid directory:
    if base_path.is_dir():
        print("Import directory looks valid!")
        #Expected foldersystem follows the format of all analysis sets created by previous scripts in the pipeline:
        images_folder = base_path / 'images'
        analysis_output_destination = base_path / 'wavefit_outputs'
        wave_analysis_outputs = analysis_output_destination / 'wave_plots'
        arcos_meta_df_folder = analysis_output_destination / 'meta_fit_saves'
        
        i += 1
    else:
        #Check if user is trying to exit the loop:
        print("This was not confirmed as a valid path...")
        print("If you want to exit the script, just hit enter!")
        if analysis_path_potential == '':
            quit()


#Load the file metadata dataframe: 
file_meta_df = pd.read_csv(base_path/'analysis_DF.csv')

#Create the wave analysis folder if it hasn't already been created. 
try:
    wave_analysis_outputs.mkdir()
except FileExistsError:
    print('Analysis wave folder already exists, proceeding.')

#Open the analysis meta dataframe:
print('Loading in ARCOS metadata frames to log circle fits...')
analysis_meta_DF_paths = arcos_meta_df_folder.glob('*_ARCOS_fit_parameters.csv')
meta_collect = []

#Collect all the logging frames, make into one and group by File ID, this is to be sure everything matches even when order is messed up. 
for dfpath in analysis_meta_DF_paths:
    meta_collect.append(pd.read_csv(dfpath, encoding='latin1'))

meta_df_build = pd.concat(meta_collect, ignore_index=True)
meta_df_build = meta_df_build[meta_df_build.columns.drop(list(meta_df_build.filter(regex='Unnamed')))]
logging_groups = meta_df_build.groupby('File_ID')
print('Logging groups built...')

#Get all the image and segmentation files: 
image_paths =  sorted(images_folder.glob('*.tif'), key=lambda x: x.stem)
arcos_dataframe_paths = sorted(analysis_output_destination.glob('*_ARCOS_df.csv'), key=lambda x: x.stem.split('_')[0])

#HAVE TO CHECK FOR WAVE DFS HERE, NEED TO RETHINK LOOP

#Load data and waveframe df
for image_path, arcos_df_path in zip(image_paths, arcos_dataframe_paths):

    #Create path to wave df file
    wave_df_path = analysis_output_destination / '{}_wave_df.csv'.format(arcos_df_path.stem.split('_')[0])

    #Check if it exists first if doing an expeditied analysis:
    if analysis_mode == 0:
        if wave_df_path.exists():
            print('Wave DF found for image {}. Skipping for expedited analysis.'.format(image_path.stem))
            continue
    
    image_data = imread(image_path)[:,0,:,:]
    print("Loaded data for image {}".format(image_path.stem))
    df_arcos = pd.read_csv(arcos_df_path)
    print('Loaded ARCOS DF #{}'.format(arcos_df_path.stem.split('_')[0]))

    wave_df_path = analysis_output_destination / '{}_wave_df.csv'.format(arcos_df_path.stem.split('_')[0])

    #Get the corresponding metadata from the master DF
    log_meta = logging_groups.get_group(int(image_path.stem))
    px_to_um_convert = log_meta.px_size.values[0]
    global_time_step = log_meta.time_step.values[0]

    #Lets check for a Wave DF:
    if wave_df_path.exists():
        print('Wave DF found! Will load this.')
        wave_df = pd.read_csv(wave_df_path, converters={"wave_centroid": read_tuple_list})
        wave_df = wave_df[wave_df.columns.drop(list(wave_df.filter(regex='Unnamed')))]
        try:
            length_of_trackable = len(wave_df['trackable'])
            print('New version DF found, moving on...')
        except:
            print('Old version DF found, adding trackable coulmn to wave DF...')
            wave_df['trackable'] = True

    else:
            #Lets do the first circle fits with a 90% fit:
            print("Running circle fits with 95 pertencile fit.")
            percentile = 95
            wave_df = calculate_wave_circle_fit(image_data, df_arcos, global_time_step, px_to_um_convert, image_path, wave_analysis_outputs, percentage=percentile)

    print("Image data and wave DF successfully loaded for Image {}".format(image_path.stem))

    #Lets loop here to give the user some control over what is going on:
    repeat_wave_analysis = ''
    continue_loop = 0

    while continue_loop == 0:

        #Open the viewer and display all waves tracked
        print("Displaying waves and data, please delete wave layers that are not accurate tracks or not waves!")
        napari_view = cleanup_wave_tracks_viewer(image_data,wave_df,TAB20)
        napari.run()

        #Collect open layers and get those groups from dataframe:
        wave_collection = []
        wave_reject_collection = []
        #NOTE THIS CODE ISNT TESTED YET HERE: 
        for layer in napari_view.layers:
            if layer.visible:
                try:
                    wave_collection.append(int(layer.name))
                except:
                    print('not wave channel...')
            else:
                wave_reject_collection.append(int(layer.name))

        if len(wave_collection) == 0:
            clTracks = wave_df.groupby('clTrackID')
            combo_waves_df = pd.concat([clTracks.get_group(j) for j in wave_reject_collection], ignore_index=True)
            combo_waves_df['trackable'] = False
        elif len(wave_reject_collection) != 0:      
            clTracks = wave_df.groupby('clTrackID')
            trackable_waves = pd.concat([clTracks.get_group(j) for j in wave_collection], ignore_index=True)
            trackable_waves['trackable'] = True
            untrackable_waves = pd.concat([clTracks.get_group(j) for j in wave_reject_collection], ignore_index=True)
            untrackable_waves['trackable'] = False
            combo_waves_df = pd.concat([trackable_waves, untrackable_waves], ignore_index=True)
        else:
            clTracks = wave_df.groupby('clTrackID')
            combo_waves_df = pd.concat([clTracks.get_group(j) for j in wave_collection], ignore_index=True)
            combo_waves_df['trackable'] = True

        ui_wave_analysis = input('Done with wave-analysis (yes,rerun,skip,split event) (y/r/n/s)?: ')
        
        if ui_wave_analysis == 'r':
            #Reloading ARCOS DF and attempting analysis:
            percentile = int(input('What percentile of data should be used to fit min-circle to?: '))
            wave_df = calculate_wave_circle_fit(image_data, df_arcos, global_time_step, px_to_um_convert, image_path, wave_analysis_outputs, percentage=percentile)

        elif ui_wave_analysis == 'y':

            #If they look good, go ahead here and append file metadata then save them to csv:
            #Should also use this space to log the parameters used in binning the calcium signal here.
            print("Resaving wave DF and corrected ARCOS DF.")
            drug_name = file_meta_df[file_meta_df['File_ID'] == int(image_path.stem)].drug_name.values[0]
            drug_dose = file_meta_df[file_meta_df['File_ID'] == int(image_path.stem)].drug_dose.values[0]
            vID = file_meta_df[file_meta_df['File_ID'] == int(image_path.stem)].volunteer_ID.values[0]
            combo_waves_df['drug_name'] = drug_name
            combo_waves_df['drug_dose'] = drug_dose
            combo_waves_df['volunteer_ID'] = vID

            df_arcos.to_csv(arcos_df_path, index=False)
            combo_waves_df.to_csv(wave_df_path, index=False)
            continue_loop = 1
            
            #Save log metadata if there is no existing log...
            temp_df_filename = arcos_meta_df_folder / (image_path.stem + '_wave_fit_log.csv')

            if temp_df_filename.exists:
                print('Fit log already exists, did not rewrite this...')
            else:
                log_meta['circle_percentage_fit'] = percentile
                print('Saving full fit parameter log DF row...')
                log_meta.to_csv(temp_df_filename)
            
        elif ui_wave_analysis == 'n':
            print('Skipping this wave set!')
            continue_loop = 1

        elif ui_wave_analysis == 's':
            #Splitting an event into two:
            cl_event_to_split = int(input('What ARCOS event needs to be split?'))
            split_tmpt = int(input('What timepoint should the split occur?'))

            #Take orignal event, then make a new event with the new clID at +1 from the max in the unique list: 
            original_clT = df_arcos[df_arcos['clTrackID']==cl_event_to_split]
            before_split_clT = original_clT[original_clT['timepoint']<=split_tmpt]
            after_split_clT = original_clT[original_clT['timepoint']>split_tmpt]
            after_split_clT = after_split_clT.assign(clTrackID = df_arcos.clTrackID.unique()[-1]+1)
            print('Split the event and appended the new two events to the dataframe, please check the results...')

            #Stitch together the dataframe and rename it as the original for the loop: 
            df_arcos = pd.concat([df_arcos[df_arcos['clTrackID']!=cl_event_to_split], before_split_clT, after_split_clT])

            #Reloading ARCOS DF and attempting analysis:
            percentile = int(input('What percentile of data should be used to fit min-circle to?: '))
            wave_df = calculate_wave_circle_fit(image_data, df_arcos, global_time_step, px_to_um_convert, image_path, wave_analysis_outputs, percentage=percentile)
                
