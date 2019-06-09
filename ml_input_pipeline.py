import numpy as np
import tensorflow as tf

class InputPipeline():    
    
  def get_special_paths(self, asset_path, asset_id, multiple_tfrecord_sites):
    paths = []
    special_site = asset_path.split("/")[3].upper()
    n_paths = [multiple_tfrecord_sites[special_site]][0]
    asset_id += "00000"
    for i in range(0,n_paths):
      patches_left = multiple_tfrecord_sites[special_site]
      n=len(str(i))
      temp_asset_id = asset_id[:-n]+str(i)
      paths.append('gs://labelled_data/'+temp_asset_id+'.tfrecord')
      multiple_tfrecord_sites[special_site] -= 1
    return paths

  
  def get_tfrecord_paths(self, fc_paths):
    subsite_filenames = [y for x in fc_paths.ALL_FILENAMES for y in x]
    input_files = []
    multiple_tfrecord_sites = {'THESSALY':2, 'IRRAWADDYDELTA':11}

    for ss in subsite_filenames:
      asset_path = ss[0]
      asset_id = asset_path.split("/")[4]

      if any(site in asset_id for site in multiple_tfrecord_sites.keys()):    
        all_tfrecord_paths_for_site = self.get_special_paths(asset_path,asset_id,multiple_tfrecord_sites)
        input_files.extend(all_tfrecord_paths_for_site)
      else:
        input_files.append('gs://labelled_data/'+asset_id+'.tfrecord')

    return input_files
  
  def make_source_dataset(self,input_files):
    target_site_code = "EMSR273"
    second_target_site_code = "EMSR258"
    source_sites = [s for s in input_files if target_site_code not in s and second_target_site_code not in s]
    print(source_sites)
    return tf.data.TFRecordDataset(source_sites)
  
  def make_target_dataset(self,input_files):
    target_site_code = "EMSR273"
    second_target_site_code = "EMSR258"
    target_site = [s for s in input_files if target_site_code in s or second_target_site_code in s]
    return tf.data.TFRecordDataset(target_site)
 
  def _parse_function(self,example_proto):
    features = {
        'VV': tf.FixedLenFeature([256,256,1], tf.float32),
        'constant': tf.FixedLenFeature([1], tf.string),
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features.pop('constant')
    image = parsed_features.pop('VV')
    raw_label = tf.decode_raw(label, tf.int8)
    raw_label_T = tf.transpose(raw_label)
    raw_reshaped_label = tf.reshape(raw_label_T, [256,256, 1])
    final_labels = tf.cast(raw_reshaped_label, tf.float32)
    return image, final_labels 
  
  
  def rotate_image(self, image, label):
    rotated_image = tf.contrib.image.rotate(image, 180)
    return rotated_image, label

  def flip_image(self, image, label):
    flipped_image = tf.image.random_flip_left_right(image)
    return flipped_image, label

  def only_with_good_ratio(self,image, label):
    is_gd = tf.greater(tf.count_nonzero(label,axis=[0,1]), tf.constant(50,tf.int64))
    return tf.reshape(is_gd,[])

  def create_one_site_only_dataset(self, input_files):
    site_code = "EMSR150"
    site = [s for s in input_files if site_code in s]
    d = tf.data.TFRecordDataset(site)
    return d.map(self._parse_function)

  def create_naive_transfer_datasets(self, input_files):
    source_d = self.make_source_dataset(input_files)
    target_d = self.make_target_dataset(input_files)
    # danger, the shuffle below takes a really long time
    train_dataset = source_d.map(self._parse_function).shuffle(5000, seed=2).skip(500) #we do this to keep the training set sizes the same between vanilla and this.
    test_dataset = target_d.map(self._parse_function).shuffle(5000, seed=2)
    return train_dataset, test_dataset  

  def create_toy_datasets(self, input_files):
    subset_input_files = input_files[0:3]
    d = self.make_source_dataset(subset_input_files)
    dataset = d.map(self._parse_function).shuffle(300, seed=2)
    train_dataset = dataset.skip(50).take(100)
    test_dataset = dataset.skip(150).take(20)
    return train_dataset, test_dataset

  def create_vanilla_datasets(self, input_files):
    d = self.make_source_dataset(input_files)
        # danger, the shuffle below takes a really long time
    dataset = d.map(self._parse_function).shuffle(5000, seed=2)
    test_dataset = dataset.take(500)
    train_dataset = dataset.skip(500)
    return train_dataset, test_dataset  
  
  def create_dataset_according_to_set_up(self,input_files, set_up):
    # Dispatcher pattern - for selecting a different set up.
    create_vanilla_datasets = lambda x: self.create_vanilla_datasets(x)
    create_naive_transfer_datasets = lambda x: self.create_naive_transfer_datasets(x)
    create_toy_datasets = lambda x: self.create_toy_datasets(x)
    dispatch = {
        'Vanilla': create_vanilla_datasets,
        'Naive Transfer': create_naive_transfer_datasets,
        'Toy': create_toy_datasets
    }
    thing = dispatch[set_up](input_files) 
    return thing

  def call(self, fc_paths, set_up='Toy'):    
    input_files = self.get_tfrecord_paths(fc_paths)
    np.random.shuffle(input_files) # RANDOMNESS introduced
    return self.create_dataset_according_to_set_up(input_files, set_up)
    #perform augmentation if desired with .map on datasets and any concat.
