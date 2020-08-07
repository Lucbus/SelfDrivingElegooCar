"""
Create folders and sort data into the right structure. 
"""
import os
import shutil
import numpy as np
from glob import glob
import operator
from pathlib import Path

class ManageData():
    def __init__(self, input_dir: Path, output_dir: Path):
        self.class_names = ['w', 'a', 's', 'd', 'x', 'e', 'q']
        self.exclusion_list = ['e', 'q']

        self.input_dir = input_dir
        self.output_dir = output_dir


    def _make_directories(self) -> None:
        """Creates train,val,test directories with subdirectories for the classes.   
        """
        print("create directories")

        Path.mkdir(self.output_dir, exist_ok=True)
        sub_directories = ['train', 'val', 'test']

        for sub in sub_directories:
            Path.mkdir(self.output_dir / sub, exist_ok=True)

        for key in self.class_names:
            if key not in self.exclusion_list:
                for sub in sub_directories:
                    Path.mkdir(self.output_dir / sub / '{}'.format(key), exist_ok=True)

    def _load_data(self) -> dict:
        """Loads raw data into dict

        Returns:
            dict: Dict containing raw data
        """        

        print("sort data into new directories")
        data = {x : [] for x in self.class_names}

        for i in self.input_dir.rglob('*.npy'):
            obs = np.load(i, allow_pickle=True).item()

            file_class = self.class_names[obs['action']-1]

            if file_class not in self.exclusion_list:

                new_file = {
                    "obs": obs['observation'],
                    "condition": obs['condition']
                }

                data[file_class].append(new_file)
        
        return data

    def _split_data(self, data: dict) -> dict:
        """Splits the data into train, test and validation set.

        Args:
            data (dict): Raw data

        Returns:
            dict: Dict of train, test and valid datasets
        """        
        print("split training set into train, test and validate")


        train_data = {x : [] for x in self.class_names}
        test_data = {x : [] for x in self.class_names}
        val_data = {x : [] for x in self.class_names}
        for dat in data.items():
            
            np.random.shuffle(np.array(dat[1]))
            train, val, test = np.split(
                dat[1], 
                indices_or_sections=[int(.80*len(dat[1])), int(.90*len(dat[1]))]
            )

            train_data[dat[0]] = train
            test_data[dat[0]] = test
            val_data[dat[0]] = val

        return {"train": train_data, "test": test_data, "val": val_data}

    def _balance_dataset(self, unbalanced_data: dict) -> dict:
        """Balances dataset by over-sampling. 

        Args:
            unbalanced_data (dict): Unbalanced dataset

        Returns:
            dict: Balanced dataset
        """               
        print("balance dataset")

        counter = {item[0]: len(item[1]) for item in unbalanced_data.items()}
        max_samples = max(counter.items(), key=operator.itemgetter(1))
        factor = 0.5
        min_samples_new = max_samples[1] * factor
        balanced_data = unbalanced_data

        for item in counter.items():
            if item[0] != max_samples[0] and item[1] !=  0:
                new_size = np.floor(min_samples_new + item[1] * (1 - factor)).astype(int)

                size_to_add = new_size - item[1]
                items_to_add = np.random.choice(unbalanced_data[item[0]], size_to_add)
                balanced_data[item[0]] = np.append(balanced_data[item[0]], items_to_add)

        return balanced_data

    def _save_data(self, data_dict: dict) -> None:
        """Saves dataset into corresponding directories

        Args:
            data_dict (dict): Data to save
        """        
        
        sub_directories = ['train', 'val', 'test']
        
        file_counter = 0

        for sub in sub_directories:
            for dat in data_dict[sub].items():
                file_path = self.output_dir / sub / dat[0]
                for fp in dat[1]:
                    output_name = file_path /  "{:05d}.npy".format(file_counter)
                    np.save(output_name, fp)
                    file_counter += 1


    def sort_dataset(self) -> None:
        """Load data from raw data folder, split data into train, val and test dataset, 
            balance train dataset and save the data into output_dir
        """        
        self._make_directories()

        data = self._load_data()

        data_sp = self._split_data(data)
        data_sp["train"] = self._balance_dataset(data_sp["train"])

        self._save_data(data_sp)


if __name__ == '__main__':
    project_path = Path.cwd().parent.parent.parent
    input_dir = project_path / 'data/raw_data' 
    output_dir = project_path / 'data/sorted_data'

    manni = ManageData(input_dir, output_dir)

    manni.sort_dataset()