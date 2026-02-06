import os
import os.path
import torch
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import json
import cv2

# TODO: Make target_field optional for unannotated datasets.
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field, bias_field=None,
                 loader=default_loader, transform=None, random_subset_size=None,
                 add_extension=None, subset=None, verbose=True):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.add_extension = add_extension
        self.subset = subset
        self.verbose = verbose
        #self.balanced_alpha = balanced_alpha
        
        self.data = pd.read_csv(csv_file)
        self.data[target_field] = self.data[target_field].replace(-1, 0).fillna(0)
        
        if bias_field is not None:
            self.bias_field = bias_field
        else:
            self.bias_field = self.target_field

        def binary_convert(x):
            if x > 0.6:
                return 1
            else:
                return 0

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()
            
        # Subset
        if self.subset is not None:
            self.data = self.data[self.data['image'].isin(self.subset)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data[self.target_field].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data[self.target_field]]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes


        ###################################
        # Create combined bias index if
        # we have multiple bias columns
        ###################################
        if isinstance(self.bias_field, list):
            # Suppose bias_field = ['bias1', 'bias2', ...]
            # Build a combined tuple column (e.g., (0,1))
            self.data['combined_bias'] = self.data[self.bias_field].apply(
                lambda row: tuple(row.values), axis=1
            )
            
            # Get all unique tuples
            unique_bias_combos = self.data['combined_bias'].unique().tolist()
            unique_bias_combos.sort()  # sorts tuples lexicographically

            # Create a mapping from each tuple to an integer
            self.bias_to_idx = {combo: idx for idx, combo in enumerate(unique_bias_combos)}
            self.biases = unique_bias_combos

            # We will refer to 'combined_bias' in the getitem
            self._actual_bias_field = 'combined_bias'
        else:
            # Single bias column
            unique_biases = self.data[self.bias_field].unique().tolist()
            unique_biases.sort()
            self.bias_to_idx = {b: i for i, b in enumerate(unique_biases)}
            self.biases = unique_biases

            self._actual_bias_field = self.bias_field

        # Print some stats if verbose
        if self.verbose:
            print('Found {} images from {} classes.'.format(len(self.data), len(classes)))
            for class_name, class_idx in self.class_to_idx.items():
                n_images = dict(self.data[self.target_field].value_counts())
                print(f"    Class '{class_name}' ({class_idx}): {n_images[class_name]} images.")
            
            # Calculate and print the amount of samples per subgroup
            print('Samples per subgroup (class and bias):')
            if isinstance(self.bias_field, list):
                # multiple bias columns
                for class_name, class_idx in self.class_to_idx.items():
                    for bias_combo in self.biases:
                        count = len(self.data[
                            (self.data[self.target_field] == class_name) &
                            (self.data['combined_bias'] == bias_combo)
                        ])
                        print(f"    Class '{class_name}' ({class_idx}), "
                              f"Bias '{bias_combo}' ({self.bias_to_idx[bias_combo]}): {count} samples")
            else:
                # single bias column
                for class_name, class_idx in self.class_to_idx.items():
                    for bias_name, bias_idx in self.bias_to_idx.items():
                        count = len(self.data[
                            (self.data[self.target_field] == class_name) &
                            (self.data[self.bias_field] == bias_name)
                        ])
                        print(f"    Class '{class_name}' ({class_idx}), "
                              f"Bias '{bias_name}' ({bias_idx}): {count} samples")
        

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field]).replace(".jpg", ".png")
        if self.add_extension:
            path = path + self.add_extension
       
        sample = self.loader(path)
             
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        group = self.bias_to_idx[self.data.loc[index, self._actual_bias_field]]
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target, group

    def __len__(self):
        return len(self.data)

class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name
        
class CSVDatasetWithCaption(CSVDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add a new column for the caption
        def create_caption(row):
            if float(row['Cardiomegaly']) == 1.0:
                lesion_type = "with enlarged heart"
            else:
                lesion_type = "with normal heart"
                
            artifacts = []
            #print(row[self.bias_field])
            if row[self.bias_field] == "Male":
                artifacts.append("man")
            elif row[self.bias_field] == "Female":
                artifacts.append("woman")
            else:
                print("There was an error. Patient gender unnavailable")
                
            caption = f"A photo of a chest x-ray of a {artifacts[0]} {lesion_type}"
            return caption

        self.data["caption"] = self.data.apply(create_caption, axis=1)

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        caption = self.data.loc[i, "caption"]
        return super().__getitem__(i), caption

class CSVDatasetWithCaptionSusu(CSVDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add a new column for the caption
        def create_caption(row):
            
            if float(row['Cardiomegaly']) == 1.0:
                lesion_type = "with enlarged heart"
            else:
                lesion_type = "with normal heart"
            
            artifacts = []
            if float(row['Contamination']) == 1.0:
                artifacts.append(" and vertical white lines")
            else:
                artifacts.append("")
                
            caption = f"A photo of a chest x-ray {lesion_type}{artifacts[0]}"
            return caption

        self.data["caption"] = self.data.apply(create_caption, axis=1)

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        caption = self.data.loc[i, "caption"]
        return super().__getitem__(i), caption

