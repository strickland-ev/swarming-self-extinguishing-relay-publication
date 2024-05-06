#Evelyn Strickland 20220601
#Script version 1.0 from jupyter notebook file importing_data.ipynb
#Goals: Take a folder location with ROIs and a meta.py file, import them into a new folder with a file ID that is 
#matched to all the metadata via DF. All analysis will be done with file-ID attached from there on out, and will be followed
#by the metadata. DF will be written out as a csv to main folder between sessions. DF also checks integrety of data it is mapped to.
#Bundled with 2022-analysis-pipeline


#System Imports:
import importlib.util
from pathlib import Path
import sys, shutil,os
from tqdm import tqdm
import copy

#Data Imports: 
import pandas as pd

#First, lets get the user to input valid filenames for import locations: 

import_Paths = []

print("Collecting import paths... Press enter when done.")
i = 0

while i < 1:
    #Take user input
    from_prompt = input("Enter import path: ")
    #Check if user is trying to exit the loop:
    if from_prompt == '':
        i += 1
        continue
    #If not, try to make the input a path:
    else:
        potential_path = Path(from_prompt)

    #Check if this path is a valid directory:
    if potential_path.is_dir():
        import_Paths.append(potential_path)
        print("Import path set.")
    else:
        print("This was not confirmed as a valid path...")

#No need to continue if no paths!
if len(import_Paths) == 0:
    print("No paths defined for import. Bye!")
    quit()

#Print the paths, and confirm with the user:
print("Are these the paths you inputed:")

for pos_path in import_Paths:
    print(pos_path)

raw_in = input("Looks good? (Y/N): ")
if raw_in != 'Y' and raw_in != 'y':
    quit()

print("Paths confirmed. Moving on.")

#Now, define a place to import to:
i = 0
while i < 1:
    #Take user input
    analysis_path_potential = input("Enter the analysis set to import into: ")
    analysis_set = Path(analysis_path_potential)

    #Check if this path is a valid directory:
    if analysis_set.is_dir():
        print("Import directory looks valid!")
        i += 1
    else:
        #Check if user is trying to exit the loop:
        print("This was not confirmed as a valid path...")
        print("If you want to exit the script, just hit enter!")
        if analysis_path_potential == '':
            i += 1

print("All set! Lets import.")
print("")

#Moving on to importing from the set paths: 

#import function built from jupyter notebook sandbox! 
def import_from_path(import_path, analysis_set):
    
    #Now, lets scan this folder for two things, number of tiff files and a metadatafile: 
    if len(sorted(import_path.glob("meta.py"))) != 1:
        print("There can only be one metadata file per folder! Please fix this before contining")
    else:
        metadata_file = sorted(import_path.glob("meta.py"))[0]
        tiff_files = sorted(import_path.glob("[!.]*.tif"))

    #If our analysis set folder is new, lets give it some structure and a Dataframe if it doesn't already have one: 
    images_path = analysis_set / 'images'
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    waves_path = analysis_set / 'wave_tracks' 
    if not os.path.exists(waves_path):
        os.makedirs(waves_path)
    segment_path = analysis_set / 'segmented_nuclei'
    if not os.path.exists(segment_path):
        os.makedirs(segment_path)

    #Now lets get the metadata imported as a module:
    #Solution from https://stackoverflow.com/questions/67631/how-do-i-import-a-module-given-the-full-path
    spec = importlib.util.spec_from_file_location("module.name", metadata_file)
    meta = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = meta
    spec.loader.exec_module(meta)

    #Now lets see if our metadata file is compatible with the destination analysis metadata dataframe
    #If there is no CSV file with an existing dataframe, lets make a new one with the first metadata file as set columns
    #Get the attributes of our importing metadata file
    meta_attrib = list(filter(lambda n: n[0] != '_', dir(meta)))

    #Path for analysis DF is fixed, if it doesn't exist yet lets make an empty one to load:
    meta_df_path = analysis_set / 'analysis_DF.csv'

    if not os.path.exists(meta_df_path):
        column_setup = ['File_ID', 'Original_File'] + meta_attrib
        meta_DF = pd.DataFrame(columns = column_setup)
        meta_DF.to_csv(meta_df_path, index=False)

    #Read in the metadata DF and set the index to the File ID:
    meta_DF = pd.read_csv(meta_df_path)

    #Next, if a dataframe exists already, lets check to make sure the columns match the incoming metadata file: 
    if list(meta_DF.columns)[1:] == meta_attrib:
        print("Metadata file compatible with this analysis set!")
    else:
        print("There might be compatibility issues with this if you proceed past this point...")

    #This is a good point to check to make sure that each entry in the DF has a corresponding file in the image folder
    #If none is found, note this to the user and delete it! 

    #See where the current image files deviate from the recorded DF from last import:
    dif_files = set(meta_DF['File_ID']).difference(set([int(n.stem) for n in list(images_path.glob('*.tif'))]))

    #If they dont deviate, good, lets continue
    if len(dif_files) == 0:
        print('DF matches existing files')
        meta_DF.reset_index()
    #If they do deviate, loop through the mismatches and delete the missing File IDs
    else:
        for i in dif_files:
            print("Dropping File_ID " + str(i) + " because it was not found.")
            meta_DF.set_index('File_ID')
            meta_DF.set_index('File_ID')
            meta_DF = meta_DF.drop(i)
            meta_DF.reset_index()

    #Save the corrected DF
    meta_DF.to_csv(meta_df_path, index=False)

    #Now that everything is setup and loaded, compatibilities checked, time to import the data and copy the images over:
    if meta_DF.empty:
        print("Starting import from file 0 since analysis_DF was empty.")
        file_ID = 0
    else:
        file_ID = meta_DF['File_ID'].max() + 1
        print("Starting import from file " + str(file_ID) + ".")
    
    #Lets now loop through each file and import it if it isn't already in the dataframe
    #Note, this will only check incoming file name, if you rename the file and try to reimport it, it will be a duplicate! 

    #Solution to get dictionary from meta module stiched together from two stackoverflows
    #https://stackoverflow.com/questions/7584418/iterate-the-classes-defined-in-a-module-imported-dynamically
    #https://stackoverflow.com/questions/9759820/how-to-get-a-list-of-variables-in-specific-python-module

    meta_dict = dict([(name, cls) for name, cls in meta.__dict__.items() if not (name.startswith('__') or name.startswith('_'))])

    for file in tqdm(tiff_files):
        new_line = copy.copy(meta_dict)
        new_line['File_ID'] = file_ID
        new_line['Original_File'] = file.name
        new_entry = pd.DataFrame(new_line, index=[file_ID])
        
        #Check to make sure this is not a duplicate file being input here: 
        if any(meta_DF['Original_File'] == new_entry['Original_File'].values[0]):
            print("Duplicate file import detected, please be more careful about adding duplicates to an analysis set")
        else:
            meta_DF = pd.concat([meta_DF,new_entry], ignore_index=True, verify_integrity = True)
            file_name = str(file_ID) + '.tif'
            dest_file = images_path / file_name
            shutil.copy(file, dest_file)
            file_ID += 1

    #Write the new DF out to memory as a csv!
    meta_DF.to_csv(meta_df_path, index=False)



#Lets import in a loop until all are covered!: 
for import_path in import_Paths:
    out = "Importing: " + str(import_path)
    print(out)
    import_from_path(import_path, analysis_set)

print("All done!")