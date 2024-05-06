#Collective Wave Identification and Cell Tracking using analysis sets, stardist labeling of neuclei, custom binning of calcium signals, and ARCOS identification for collective events: 
#Written: Evelyn Strickland, 2022
#Script Ver: 1.0
#bundled with 2022-analysis-pipeline

#Changelog:
#Cleaned and split up pipeline from run_arcos_set v0.1 so that wave circle fitting happens in next script, too bulky to be used in this loop.

#------------------------------------------------------------------------------------------------------------------------
#Importing section, Function setup: 

#Basic tooling, tracking:
from pathlib import Path
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import trackpy
from tqdm import tqdm
from matplotlib import pyplot as plt

#Image Tooling: 
from skimage.io import imread
from skimage.measure import regionprops_table
from skimage.util import map_array

#Napari and tools
import napari
from qtpy.QtCore import QTimer
#from naparimovie import Movie

#ARCOS Tools: 
from arcos4py import ARCOS
from arcos4py.tools import filterCollev 

#Convienient colormap wrapper for napari ARCOS display: 
TAB20 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",]

#------------------------------------------------------------------------------------------------------------------------
#ARCOS specific function from:
#https://arcos.gitbook.io/home/example-use-cases/detecting-collective-signalling-events-in-epithelial-cells#analyse-data-with-arcos
def remap_segmentation(df: pd.DataFrame, segmentation: list, timepoint_column: str = 'timepoint', label_column: str = 'label', measure_column: str = 'Calcium') -> list:
    tracked_numpy = df[[timepoint_column, label_column, measure_column]].sort_values(timepoint_column).to_numpy()
    grouped_numpy = np.split(tracked_numpy,np.unique(tracked_numpy[:,0], return_index = True)[1][1:])
    ratio_remapped = []
    for img, grp in zip(segmentation, grouped_numpy):
        img_copy = map_array(img, grp[:,1], grp[:, 2])
        ratio_remapped.append(img_copy)
    return ratio_remapped


#Pipeline functions:
def image_tracking(image_data,segmentation,px_to_um_convert):
    
    #Takes in an image, segmentation image, pulls label information and tracks the points using trackpy, filters the tracks and formats for use in the ARCOS algorithm.
    #Returns a pandas dataframe with rows being cell points with particle IDS as tracked, formatting ready for binarization of the calcium signal
    
    df = []
    trackpy.quiet(suppress=True)
    
    #Look over the image data and pull the labels out of the segmentaion image, use these to build a dataframe with the centroid, intensity from the calcium channel, area of track spot, label, etc.
    print("Collecting labels from segmented image...")
    pbar = tqdm(total=len(image_data))
    for t, img_frame in enumerate(image_data):
        '''
        if t%10 == 0:
            print(f'analysing timepoint {t}')
        '''
        labels = segmentation[t]
        dic = regionprops_table(labels, img_frame, properties=['label', 'centroid', 'intensity_mean', 'area'])
        dic['timepoint'] = np.repeat(t, len(dic['label']))
        
        #Calculate the density of cells per frame and put in DF: 
        #This calculation should be taken out of the loop? Frame size doesn't change over time
        frame_area = (2*px_to_um_convert*img_frame.shape[0]*img_frame.shape[1])/1000000.0 #Note: This is px * px, to px^2 --> um^2 then taken to --> mm^s

        density_frame = (len(np.unique(labels))-1)/frame_area #Density in cells/mm^2
        dic['cell_density'] = np.repeat(density_frame, len(dic['label']))
        
        df.append(pd.DataFrame(dic))
        pbar.update(1)
    
    #Build out the dataframe, perform some formatting for ARCOS, then link the points using trackpy
    print('Tracking cells from segmented image.')
    df_full = pd.concat(df)
    #ARCOS and trackpy are picky about what they want columns to be called, namely the x and y coordinate of a cell
    df_full = df_full.rename(columns={"centroid-1": "x", "centroid-0": "y", 'intensity_mean': 'Calcium'})
    df_full = df_full.sort_values(['timepoint'])
    df_tracked = trackpy.link_df(df_full, neighbor_strategy='KDTree', link_strategy='auto', search_range = 3, adaptive_stop=2, memory = 3, t_column = 'timepoint') #These defaults yeild decent performance without breaking on all the images I use
    df_tracked = df_tracked.reset_index(drop=True).rename(columns={'particle': "track_id"})
    
    
    #Filter the tracks to remove small track stubs!
    filt_tracks = []
    df_grouped = df_tracked.groupby('track_id')

    for i, track in df_grouped:
        if len(track) >= 15:
            filt_tracks.append(track)

    return pd.concat(filt_tracks, ignore_index=True)

def custom_bin_arcos(df_filter_track, neighbor_dist=25, upper_mean_clip=3500, deviation_factor = 1.15, min_cluster = 15, min_collective_duration = 6, min_collective_size = 50):
    
    #Notes:
    
    #This will take a dataframe built by the image_tracking function and binarize the calcium coulmn by looking for places where the calcium signal is some percentage above 
    #its global mean. This generally seperates on from off states for decent enough tracks. 
    #It will then run the ARCOS algorithm and group collective events into the output dataframe and the input binarized dataframe just in case: 
    
    #deviation_factor:
    #A signal must deviate above its mean by this amount (eg 1.1 = 10% above mean) for it to be considered 'on'
    #This works fairly well for calcium pulses
    
    #upper_mean_clip:
    #clip signals that have means far above what normal activation states would be (dying cells?) 
    #Above this limit, tracks are simply ignored and given no 'on' states
    
    print('Binning calcium signals...')
    filtered_groups = df_filter_track.groupby('track_id')
    group_collect = []

    #NOTEIMPROVEMENT: Could add some multithreading here to speed this part up...
    for i, group in filtered_groups:
        cal_val = group.Calcium.values
        bin_val = np.zeros(len(cal_val), dtype='int8')

        if np.mean(cal_val) >= upper_mean_clip:
            continue
        else:
            for indx, val in enumerate(cal_val):
                if val > np.mean(cal_val)*deviation_factor:
                    bin_val[indx] = int(1)

        group['Calcium.bin'] = bin_val
        group_collect.append(group)

    '''
    print('Optomizing neighbor distance...')
    

    cell_nums = []
    frame_track = df_bin_filt_track.groupby('timepoint')
    for i, frame in frame_track:
        cell_nums.append(len(frame))
    neighbor_dist = round(image_size/np.mean(cell_nums)-5)
    '''
    
    
    print('Running ARCOS...')
    df_bin_filt_track = pd.concat(group_collect, ignore_index=True)
    ts = ARCOS(df_bin_filt_track, ["x", "y"], 'timepoint','track_id', 'Calcium')
    ts.bin_meas = "Calcium.bin"
    df_arcos = ts.trackCollev(eps=neighbor_dist, minClsz=min_cluster)
    
    #Filter the collective events to clean up suprious collective events: 
    filterer = filterCollev(df_arcos, 'timepoint', 'clTrackID', 'track_id')
    ts_filt = filterer.filter(coll_duration = min_collective_duration, coll_total_size = min_collective_size)
    
    return (df_bin_filt_track, ts_filt)
    
def napari_view_arcos(image, df_tracked, segmentation_data, ts_filtered, color_map_custom):
    print('Setting up napari view data...')
    #np_data = df_tracked[['track_id', 'timepoint', 'y', 'x']].to_numpy()
    colors = np.take(np.array(color_map_custom), ts_filtered['clTrackID'].unique(), mode="wrap")
    df_w_colors = pd.merge(ts_filtered, pd.DataFrame(data={'colors': colors, 'clTrackID': ts_filtered['clTrackID'].unique()}))
    points_data = df_w_colors[['timepoint', 'y', 'x']].to_numpy()
    #colors_data = df_w_colors['colors'].to_numpy('str')
    #ratio_remapped = remap_segmentation(df_tracked, segmentation_data)
    #ratio_remapped = np.stack(ratio_remapped)
    
    #Setup and display the viewer:
    
    viewer = napari.Viewer()
    viewer.add_image(image, name='Calcium image', colormap='inferno')
    #viewer.add_image(ratio_remapped, colormap='viridis')
    #viewer.add_labels(np.stack(segmentation_data), name='segmentation', visible=False)
    #viewer.add_tracks(np_data, name='cell tracks')
    viewer.add_points(points_data, face_color=colors_data, name='collective events')
    napari.run()

    del viewer
    
    #Movie functionality just in case: 
    #movie = Movie(myviewer=viewer)
    #movie.make_movie(name = 'movie.mp4', resolution = 200, fps = 22)

def napari_view_arcos_basic(image, ts_filtered, color_map_custom):
    print('Setting up napari view data...')
    colors = np.take(np.array(color_map_custom), ts_filtered['clTrackID'].unique(), mode="wrap")
    df_w_colors = pd.merge(ts_filtered, pd.DataFrame(data={'colors': colors, 'clTrackID': ts_filtered['clTrackID'].unique()}))
    points_data = df_w_colors[['timepoint', 'y', 'x']].to_numpy()
    colors_data = df_w_colors['colors'].to_numpy('str')
    contrast_list = [np.min(image), np.max(image)*0.1]
    
    #Setup and display the viewer:
    
    viewer = napari.Viewer()
    viewer.add_image(image, name='Calcium image', contrast_limits=contrast_list, colormap='gray')
    viewer.add_points(points_data, edge_color=colors_data, face_color=[0,0,0,0], opacity=0.4, name='collective events')
    napari.run()
    del viewer
    

def arcos_summary_plot(image_path, image_data, segmentation_data, df_arcos, colormap, save_path):
    cal_sum = np.array([np.sum(image_data[i]) for i in range(len(image_data))])
    n_cells_frame = np.array([len(np.unique(segmentation_data[i]))-1 for i in range(len(segmentation_data))])
    clevents = df_arcos.groupby('clTrackID')
    clIDs = df_arcos['clTrackID'].unique()
    colors = np.random.choice(colormap, size=len(clIDs))
    start_stops = [(clevents.get_group(i).timepoint.values[0], clevents.get_group(i).timepoint.values[-1]) for i in clIDs]

    fig, ax1 = plt.subplots(figsize=(15,15))

    ax2 = ax1.twinx()
    ax1.plot(cal_sum, 'b')
    ax2.plot(n_cells_frame, 'r')

    for clID, clr, strstp in zip(clIDs, colors, start_stops):
        ax1.text(strstp[0]-5,cal_sum[strstp[0]]*1.01, str(clID), c=clr)
        ax1.fill_betweenx(cal_sum, strstp[0], strstp[1], alpha=0.6, color=clr)

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Normed Sum Calcium Signal', color='b')
    ax2.set_ylabel('Normed Cell Count', color='r')
    ax1.set_ylim(np.percentile(cal_sum, 5),np.percentile(cal_sum, 100)*1.03)
    ax2.set_ylim(np.percentile(n_cells_frame, 10),np.percentile(n_cells_frame, 100)*1.5)
    plt.title('ARCOS Anlaysis Summary for Image {}:'.format(image_path.stem))
    fig.savefig(save_path / (image_path.stem + '_ARCOS_summary.png'), dpi=200)
    plt.close(fig)


#------------------------------------------------------------------------------------------------------------------------
#Run code below here: 
#Note, this will mainly be just pointed at analysis set folders, then do its thing in tracking the waves found in each file in the set.
#At each point, a napari viewer will display how the analysis is doing before ellipse fitting, and the user can cancel an analysis. 
#Will output how elipse fitting looks, output wave results with a df formatted csv, and a dataframe formatted csv with the tracking data + binarized calcium, etc.


#Planned Updates: (Search for NOTEFORIMPROVEMENT)
#Batching of trackpy code, this could be a lot faster
#For now, runs through whole set. In future will check if analysis already performed, then ask if user wants to re-run it. 
#Will ask if user wants to manually adjust any parameters before running analysis. 
#Will also output a graph showing whole calcium signal, then annotate sections where ARCOS as determined an event.


#Now, define a place to import to:
i = 0
while i < 1:
    #Take user input
    #Do you want to run a full analysis or expedited? (f/e):
    mode_query = input("Do you want to run a full analysis,expedited, or tracking only? (f/e/t): ")
    if mode_query == 'e':
        analysis_mode = 0
        print('Mode set to expidited.')
    elif mode_query == 't':
        analysis_mode = 2
    else:
        analysis_mode = 1
        print('Mode set to full.')

    analysis_path_potential = input("Enter the set to analyze: ")
    base_path = Path(analysis_path_potential)

    #Check if this path is a valid directory:
    if base_path.is_dir():
        print("Import directory looks valid!")
        images_path = base_path / 'images'
        segmentation_path = base_path / 'segmented_nuclei'
        analysis_output_destination = base_path / 'wavefit_outputs'
        'Creating output folder...'
        try:
            analysis_output_destination.mkdir()
        except FileExistsError: 
            'Output path already exists, proceeding anyway.'
        i += 1
    else:
        #Check if user is trying to exit the loop:
        print("This was not confirmed as a valid path...")
        print("If you want to exit the script, just hit enter!")
        if analysis_path_potential == '':
            quit()


#Need to save meta fit data even in event of crash or force quit temporarily somewhere:
temp_meta_destitation = analysis_output_destination / 'meta_fit_saves'
try:
    temp_meta_destitation.mkdir()
except FileExistsError: 
    'Temp save location already exists, continuing... '


#Get all the image and segmentation files: 
images =  sorted(images_path.glob('*.tif'), key=lambda x: x.stem)
segmentations = sorted(segmentation_path.glob('*.tif'), key=lambda x: x.stem.split('_')[0])

#Check to see if the number of images match the number of segmentations: 
#(Lazy way to do this but idk, we trust the inport because of the way sets are established)
if len(images) != len(segmentations):
    print("The number of images and segmentations is different! Check these folders to make sure each image has its segmentation already performed.")
    quit()

#Open the analysis meta dataframe:
print('Loading in analysis set metadata DF.')
analysis_meta_DF = pd.read_csv(base_path / 'analysis_DF.csv')
file_meta_groups = analysis_meta_DF.groupby('File_ID')
meta_collect = []

#Time to run the main loop of code, then save the output image sets and the associated dataframes! 
for image, segmentation in zip(images,segmentations):
    #Read in images. 
    #(To be fair, this could be more efficent using pages and loading only the needed part, I am being lazy here...)
    image_data = imread(image)[:,0,:,:]
    image_size = image_data.shape[1] * image_data.shape[2]
    segmentation_data = imread(segmentation)

    #Get the corresponding metadata from the master DF
    try: 
        import_meta = file_meta_groups.get_group(int(image.stem))
        px_to_um_convert = import_meta.px_size.values[0]
        global_time_step = import_meta.time_step.values[0]
    except:
        print('Mismatch with file and metaDF????')
        continue

    #Check if image data matches shape of segmentation data, and run analysis
    if image_data.shape == segmentation_data.shape:
        print('Running analysis for: ' + image.name + ' with segmentation ' + segmentation.name + '.')
        #Basic run through for a single image:

        #Lets check to see if the tracking DF and ARCOS DF already exist, if you want to recalculate them simply delete them.
        tracking_df_path = analysis_output_destination / (image.stem + '_tracking_df.csv')
        tracking_df_binned_path = analysis_output_destination / (image.stem + '_tracking_binned_df.csv')
        arcos_df_path = analysis_output_destination / (image.stem + '_ARCOS_df.csv')

        #Check the  mode for the program:
        #If zero, this means full analysis has been turned off, and images with a tracking, binned tracking, and arcos dataframe will 
        #be skipped...
        if analysis_mode == 0:
            if np.all([tracking_df_path.exists(), tracking_df_binned_path.exists(), arcos_df_path.exists()]):
                print('All DFs found, skipping image {}...'.format(image.stem))
                continue
        
        #If 2, then we only want to build a tracking file. For large datasets and high densities, the tracking part of the loop is very time consuming, and there is nothing to adjust. It can be good to start the analysis by doing all the 
        #tracking at once then going back and re-runing the program just to force it to bin and run arcos tuning steps seperately. 
        if analysis_mode == 2:
            if tracking_df_path.exists():
                print("Tracking file for image {} found, skipping tracking step...".format(image.stem))
                continue
            else: 
                tracking_df = image_tracking(image_data,segmentation_data, px_to_um_convert)
                tracking_df.to_csv(tracking_df_path, index=False)
                print("Wrote out just the tracking file, skipping ARCOS and napari tuning. re-run in a different mode to activate these features...")
                continue

        #Setting default values:
        continue_loop = 0
        neighborhood_distance = 25 #Default value
        bin_threshold_percent = 1.15 #Defaults

        #If they exist, load them into memory from csvs.
        if tracking_df_path.exists():
            print('Tracking DF found already, using this instead of recalculating.')
            tracking_df = pd.read_csv(tracking_df_path)
            if np.all([tracking_df_binned_path.exists(), arcos_df_path.exists()]):
                print('ARCOS DF and binned tracking DF found already, using this instead of recalculating.')
                df_arcos = pd.read_csv(arcos_df_path)
                tracking_df_binned = pd.read_csv(tracking_df_binned_path)
                need_first = False
            else: 
                print('ARCOS files not found...')
                need_first = True

        else:
            #Otherwise, track and binarize calcium signals, then input it into ARCOS and display the results on napari.
            print('Tracking cells and collecting calcium traces.')
            tracking_df = image_tracking(image_data,segmentation_data, px_to_um_convert)
            tracking_df.to_csv(tracking_df_path, index=False)
            need_first = True
            #Should log if tracking had to reduce search tree here.

        #Lets loop here to give the user some control over what is going on:
        continue_wave_analysis = ''

        while continue_loop == 0:

            if need_first:
                tracking_df_binned, df_arcos = custom_bin_arcos(tracking_df, neighbor_dist=neighborhood_distance, deviation_factor=bin_threshold_percent)
                tracking_df_binned.to_csv(tracking_df_binned_path, index=False)
                #df_arcos.to_csv(arcos_df_path, index=False)

            while len(df_arcos) == 0:
                print('ARCOS has found no collective events. Please check csvs for obvious issues in tracking or binning. If things look good lets try a looser ARCOS...:')
                print('Please adjust the neighbor distance (25):')
                neighborhood_distance = int(input('Neighborhood to try:'))
                print('Please adjust the bin-threshold percent (1.15):')
                bin_threshold_percent = float(input('Threshold to try:'))
                tracking_df_binned, df_arcos = custom_bin_arcos(tracking_df, neighbor_dist=neighborhood_distance, deviation_factor=bin_threshold_percent)
                tracking_df_binned.to_csv(tracking_df_binned_path, index=False)
                #df_arcos.to_csv(arcos_df_path, index=False)
                print('saved results, checking if length still zero...')
            
            print('Collective events detected!')
            print('Please check the napari output to determine how ARCOS has done so far.')
            #napari_view_arcos(image_data, tracking_df_binned, segmentation_data, df_arcos, TAB20)
            napari_view_arcos_basic(image_data, df_arcos, TAB20)
            
            continue_wave_analysis = input('Did the analysis look good enough to continue(y), repeat (r), or skip(n) (y/n/r): ')
            
            if continue_wave_analysis == 'r':
                print('Defaults in ()!')
                print('Please adjust the neighbor distance (25):')
                neighborhood_distance = int(input('Neighborhood to try:'))
                print('Please adjust the bin-threshold percent (1.15):')
                bin_threshold_percent = float(input('Threshold to try:'))
                tracking_df_binned, df_arcos = custom_bin_arcos(tracking_df, neighbor_dist=neighborhood_distance, deviation_factor=bin_threshold_percent)
                need_first = False

            elif continue_wave_analysis == 'n':
                print('Continuing without saving anything...')
                continue_loop = 1
            elif continue_wave_analysis == 'y':
                #If they look good, go ahead here and save them to csv:
                #Should also use this space to log the parameters used in binning the calcium signal here.
                print('Outputing binarized tracking and ARCOS dataframes.')
                tracking_df_binned.to_csv(tracking_df_binned_path, index=False)
                df_arcos.to_csv(arcos_df_path, index=False)
                print('Plotting ARCOS summary...')
                arcos_summary_plot(image, image_data, segmentation_data, df_arcos, TAB20, analysis_output_destination)
                
                #Need to supress an error here? 
                temp_df_filename = temp_meta_destitation / (image.stem + '_ARCOS_fit_parameters.csv')

                if temp_df_filename.exists():
                    continue_loop = 1
                    print('Since log exists, did not overwrite log data...')
                else:
                    print('Saving image fit metadata...')
                    import_meta['arcos_neighbor_dist'] = neighborhood_distance
                    import_meta['calcium_bin_threshold'] = bin_threshold_percent
                    import_meta.to_csv(temp_df_filename, index=False)
                    continue_loop = 1

        #Clean up variables here?
        print('Cleanup...')
        del image_data
        del segmentation_data
        del tracking_df_binned
        del tracking_df
        del df_arcos
        del import_meta

    else:
        print('Image ' + str(image.name) + ' doesnt match its segmentation, please check this.')
        continue

print('No more files in analysis DF to analyze!')
