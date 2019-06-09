import pandas as pd
import datetime
import ee

# This module knows how to make DIs and how to export them to GCS as GeoTIFFs. 
# It uses temporal info from a csv of subsite metadata to make mosaics of the correct timings.
# It uses pandas dataframes as bread and butter

def make_DIs(subsites_df, clipped_collection_VV):
    diff_imgs = []
    for i, row in subsites_df.iterrows():
        before_date, during_date = row['Before Flood'], row['Map date']
        before_range = (ee.Date(str(before_date)), ee.Date(str(before_date + datetime.timedelta(days=19))))
        after_range = (ee.Date(str(during_date-datetime.timedelta(days=10))), ee.Date(str(during_date+datetime.timedelta(days=10))))
        before = clipped_collection_VV.filterDate(before_range[0], before_range[1]).mosaic()
        during = clipped_collection_VV.filterDate(after_range[0],after_range[1]).mosaic()
        diff_imgs.append([row['Subsites'], during.subtract(before) ])
    new_df = pd.DataFrame(diff_imgs, columns=['Subsites', 'Difference Image'])
    subsites_df = subsites_df.merge(new_df, on='Subsites')
    return subsites_df

def get_subsites_df(file_name):
    subsites_df = pd.read_csv(file_name) 
    subsites_df = subsites_df[pd.notnull(subsites_df['Map date'])]
    subsites_df['Subsites'] = subsites_df['Subsites'].str.replace(" ","")
    return subsites_df

def add_temporal_info(subsites_df):
    subsites_df['Map date'] = subsites_df['Map date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y').date(), na_action='ignore')
    subsites_df['Before Flood'] = subsites_df['Map date'].map(lambda x: (x - datetime.timedelta(weeks=16)), na_action='ignore') # see unresolved issues, some sites in myanmar got 15 week timedelta and everyone else got 8 :/ it was me debugging and this was not the solution so this is kind of tech debt that affects the scientificness of the overall study TODO.
    return subsites_df

def add_spatial_info(subsites_df, poly_df):
    # geom_df = pd.Dataframe.from_dict(rois_by_ss, columns=['Subsites','Geometry'])
    subsites_df = subsites_df.merge(poly_df, on='Subsites')
    return subsites_df

def export_to_gcs(image,geom,file_name_prefix):
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        bucket='diff_imgs',
        fileFormat='GeoTIFF',
        fileNamePrefix=file_name_prefix,
        scale=10,
        region=geom,
        cloudOptimized=True
    )
    task.start()

def export_all_DIs_to_gcs(subsites_df_with_DIs):
    for row in subsites_df_with_DIs:
        geom, di, subsite_name = subsites_df_with_DIs['Geometry'], subsites_df_with_DIs['Difference Image'], subsites_df['Subsites']
        export_to_gcs(di, geom.geometry().bounds().getInfo()["coordinates"], subsite_name)

def add_DIs(poly_df, clipped_collection_VV):
    subsites_df = get_subsites_df('subsites.csv')
    subsites_df = add_temporal_info(subsites_df)
    subsites_df = add_spatial_info(subsites_df, poly_df)
    subsites_df_with_DIs = make_DIs(subsites_df, clipped_collection_VV)
    # export_all_DIs_to_gcs(subsites_df_with_DIs)
    return subsites_df_with_DIs
