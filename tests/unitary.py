from HC4CA.classes import Dataset

nb_dir = "Dataset/"
meta_path = nb_dir + "/metadata/"
sphere_path = nb_dir

sphere = Dataset(sphere_path, ["train", ],
                 sensors=['acceleration', 'pir', 'video'],
                 meta_path=meta_path)

train_data = sphere.data['train'].raw_data.copy()
print(train_data.head())
