import pandas as pd
import os
import json
import pickle
import copy
from datetime import datetime


class Dataset(object):
    def __init__(self,
                 data_path,
                 subsets,
                 sensors,
                 **kwargs):
        """

        :param data_path:
        :param subsets:
        :param sensors:
        :param kwargs:
        """
        self.meta_path = kwargs.pop('meta_path', None)

        self.metadata = None
        self.data = None
        self.data_path = data_path
        self.sensors = sensors
        self.subsets = subsets

        self.load()

    def _load_metadata(self):
        # read metadata files
        if self.meta_path is not None:
            metadata = self._read_metadata_files()
        else:
            metadata = None

        return metadata

    def _read_metadata_files(self):
        # read json files and return a dict with them
        metadata = {}
        for file in os.listdir(self.meta_path):
            filename = f"{self.meta_path}/{file}"
            meta = file.split('.')[0]
            metadata[meta] = json.load(open(filename))
        return metadata

    def _load_data_subsets(self):
        # populates data dict with read subsets
        data = {}
        for subset in self.subsets:
            data[subset] = DataSubset(subset, self)
        return data

    def load(self):
        self.metadata = self._load_metadata()
        self.data = self._load_data_subsets()

    def resample(self, inplace=True, **kwargs):
        freq = kwargs.pop('freq', '1S')

        # resample each subset in the dataset
        for subset in self.subsets:
            resampled_data = self.data[subset].resample(freq=freq)

            # resample locations if there is one
            resampled_locations = self.data[subset].resample_location(freq=freq)
        if inplace:
            self.data[subset].raw_data = resampled_data
            self.data[subset].locations = resampled_locations

        else:
            new_obj = copy.deepcopy(self)
            new_obj.data[subset].raw_data = resampled_data
            new_obj.data[subset].locations = resampled_locations
            return new_obj

    def to_csv(self):  # TODO: prepare to csv
        return self

    def dump(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        file_prefix = kwargs.pop("file_prefix", "DatasetObj_")
        time_stamp = kwargs.pop('time_stamp', True)

        if time_stamp:
            time = datetime.now().strftime("%H%M%S%d%m%y")
            file_prefix = file_prefix + time

        file = file_prefix + ".pkl"

        with open(file, "wb") as f:
            pickle.dump(self, f)
        return file

    @staticmethod
    def load_from_pickle(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj


class DataSubset(object):
    def __init__(self, subset, dataset):
        self.subset = subset  # name i.e. "train"
        self.dataset = dataset  # parent dataset
        self.subjects = None
        self.targets = None
        self.raw_data = None
        self.locations = None

        self.load()

    def __str__(self):
        return f"{self.raw_data}"

    def _repr_html_(self):
        return self.raw_data._repr_html_()

    def __repr__(self):
        return self.__str__()  # repr(self.raw_data)

    def _get_subjects(self):
        # returns a list of subjects objects found
        list_subjects = []
        directory = f"{self.dataset.data_path}" \
                    f"/{self.subset}"
        subjects = os.listdir(directory)

        for subject in subjects:
            subject = str(subject).zfill(5)  # TODO: to parameter
            obj_subject = Subject(subject, self)
            list_subjects.append(obj_subject)
        return list_subjects

    def _has_targets(self):
        # check whether subset has targets.
        targets = False
        if self.subset == 'train':
            targets = True
        return targets

    def _get_raw_data(self):
        # returns read data per each subject
        data = pd.DataFrame([])
        for obj_subject in self.subjects:
            read_data = obj_subject.read_data()
            data = pd.concat([data, read_data])

        return data.sort_index()

    def _get_locations(self):
        """
        Obtains location labels for each subject in the dataset.

        :return:
        if targets available, returns locations annotated for subject in a dataframe
        else returns an empty dataframe.
        """
        locations = pd.DataFrame([])

        if self.targets:
            for obj_subject in self.subjects:
                read_data = obj_subject.read_location()
                locations = pd.concat([locations, read_data])

        return locations.sort_index()

    def _add_no_label(self, **kwargs):
        no_label_value = kwargs.pop('no_label_value', 0.0)
        return self.locations.copy().insert(0, ('Locations', 'no_label'),
                                            no_label_value)

    def load(self):
        self.subjects = self._get_subjects()
        self.targets = self._has_targets()  # TODO: targets only for train subset
        self.raw_data = self._get_raw_data()
        self.locations = self._get_locations()

    def get_location_labels(self, no_label=False,
                            annotated_only=True,
                            **kwargs):
        if self.locations.empty:
            raise ValueError("subset doesn't have location data")

        if no_label:
            df = self._add_no_label(**kwargs)
        else:
            df = self.locations.copy()

        if annotated_only:
            df = df[df.sum(axis=1) > 0]

        # get rid of multi-index to get just column name
        df.columns = df.columns.droplevel(0)

        return df.idxmax(axis=1)

    # TODO: quick fix
    def resample_location(self, **kwargs):
        freq = kwargs.pop('freq', '1S')

        if self.locations is None:
            return None

        df1 = pd.DataFrame([])
        # resampling doesn't work on multi-index.
        # we get each value at level 0 and then recreate the index
        for subject in self.locations.index.levels[0]:
            df = self.locations.loc[subject]
            df = df.resample(freq).mean()
            df.index = pd.MultiIndex.from_product([[subject], df.index])
            df1 = pd.concat([df1, df])
        return df1

    # TODO: write resample with  in_place=False, **kwargs
    def resample(self, **kwargs):
        """
        Change sample rate to freq (default 1 sec)
        :param kwargs:
        :return:
        """
        freq = kwargs.pop('freq', '1S')
        # func = kwargs.pop('func', pd.core.groupby.generic.DataFrameGroupBy.mean)

        groupby_levels = [0]  # Level to group by
        resample_level = 1  # Level where datetime-like index is
        level_values = self.raw_data.index.get_level_values
        grouped = (self.raw_data.groupby([level_values(i) for i in groupby_levels]
                                         + [pd.Grouper(freq=freq, level=resample_level)]))

        return grouped.mean()

    # Always adds locations if they exists
    def to_pickle(self, **kwargs):
        sensors = kwargs.pop('sensors', None)
        file_prefix = kwargs.pop("file_prefix", "DatasubsetObj_")
        time_stamp = kwargs.pop('time_stamp', True)

        # set output file name
        file = file_prefix + f'_{self.subset}_'
        if time_stamp:
            time = datetime.now().strftime("%H%M%S%d%m%y")
            file = file + time
        file = file + ".gzip"

        df = self.raw_data.copy()

        # separating sensors
        df = self.get_only(df, sensors=sensors)

        # add locations if they exists in the subset
        if self.locations is not None:
            df = pd.concat([df, self.locations], axis=1)

        df.to_pickle(file, compression='gzip')

        return file

    @staticmethod
    def read_pickle(filename):
        return pd.read_pickle(filename, compression='gzip')

    @staticmethod
    def read_csv(filename):
        return pd.read_csv(filename)

    @staticmethod
    def get_only(df, sensors):
        if sensors == 'all':
            return df
        df_sensors = set(df.columns.get_level_values(0))
        for sensor in sensors:
            if sensor not in df_sensors:
                raise ValueError(sensor + " not in dataframe")

        return df.loc[(slice(None), sensors)]


class Subject(object):
    def __init__(self, subject, datasubset):
        self.datasubset = datasubset  # subject's subset
        self.subject = subject  # name i.e. "00001"
        self.path = f"{datasubset.dataset.data_path}/" \
                    f"{datasubset.subset}/" \
                    f"{subject}"

        self.metadata, self.start, self.end, self.annotators = \
            None, None, None, None
        self._index = None

        self.load()

    def _get_meta(self):
        filename = f"{self.path}/meta.json"

        if Subject._file_exists(filename):
            meta = json.load(open(filename))
            start = meta['start']
            end = meta['end']
            annotators = meta['annotators']
            metadata = True
        else:
            metadata = start = end = annotators = None

        return metadata, start, end, annotators

    def _get_index(self, **kwargs):
        freq = kwargs.pop("freq", '50L')
        unit = kwargs.pop("unit", 'S')
        index = None
        if self.metadata:
            index = pd.timedelta_range(self.start, freq=freq,
                                       end=str(self.end + 1) + unit)
        return index

    def _read_pir_old(self, **kwargs):
        ext = kwargs.pop("ext", 'csv')
        sensor = 'pir'
        filename = f"{self.path}/{sensor}.{ext}"

        df = pd.read_csv(filename, na_filter=True)
        # Dealing with indexes
        df.start = pd.TimedeltaIndex(df.start, unit="s", name="start")
        df.end = pd.TimedeltaIndex(df.end, unit="s", name="end")

        df = self._set_multi_index(df, sensor)

        return df

    def _read_pir(self, **kwargs):
        ext = kwargs.pop("ext", 'csv')
        sensor = kwargs.pop("sensor", 'pir')
        filename = f"{self.path}/{sensor}.{ext}"
        pir_loc = self.datasubset.dataset.metadata['pir_locations']

        df = pd.read_csv(filename, na_filter=True)
        # Dealing with indexes
        df.start = pd.TimedeltaIndex(df.start, unit="s", name="start")
        df.end = pd.TimedeltaIndex(df.end, unit="s", name="end")

        df = self._read_by_interval(df, columns=pir_loc)
        df = self._set_multi_index(df, sensor)

        return df

    def _read_by_interval(self, df, **kwargs):
        freq = kwargs.pop("freq", '50L')
        unit = kwargs.pop("unit", 'S')
        columns = kwargs.pop("columns", None)
        min_index = self._index

        if min_index is None:
            min_index = pd.timedelta_range(df.start.min(), freq=freq,
                                           end=str(df.end.max() + 1) + unit)

        df_t = pd.DataFrame(0, index=min_index, columns=columns)
        lower = df_t.index[0]

        for upper in df_t.index:
            interval = df[(df.start < upper) & (df.end >= lower)]
            if interval.size != 0:
                for ir, row in interval.iterrows():
                    df_t[row[2]][lower] = 1
            lower = upper

        return df_t

    def _read_acceleration(self, **kwargs):
        ext = kwargs.pop("ext", 'csv')
        sep_rssi = kwargs.pop("sep_rssi", True)
        # freq = kwargs.pop("freq", '50L')
        unit = kwargs.pop("unit", 'S')
        round_freq = kwargs.pop("round_freq", 'L')
        sensor = kwargs.pop('sensor', 'acceleration')

        filename = f"{self.path}/{sensor}.{ext}"
        df = pd.read_csv(filename, index_col="t", parse_dates=True, na_filter=True)
        df.index = pd.TimedeltaIndex(df.index, unit=unit, name="t",
                                     freq='infer').round(freq=round_freq)

        df = self._set_multi_index(df, sensor)
        if sep_rssi:  # TODO: change how rssi is separated
            df.columns = pd.MultiIndex.from_tuples([('rssi', 'Kitchen_AP'),
                                                    ('rssi', 'Lounge_AP'),
                                                    ('rssi', 'Study_AP'),
                                                    ('rssi', 'Upstairs_AP'),
                                                    ('acceleration', 'x'),
                                                    ('acceleration', 'y'),
                                                    ('acceleration', 'z')])

        return df

    def _read_video(self, **kwargs):
        ext = kwargs.pop("ext", 'csv')
        sensor = kwargs.pop("sensor", 'video')
        # ['hallway', 'kitchen', 'living_room', ])
        vid_loc = self.datasubset.dataset.metadata['video_locations']

        df = pd.DataFrame([])
        for loc in vid_loc:
            filename = f"{self.path}/{sensor}_{loc}.{ext}"
            df_v = pd.read_csv(filename, index_col="t", parse_dates=True,
                               na_filter=True,
                               )
            df_v.index = pd.TimedeltaIndex(df_v.index, unit="s", name="t")
            df_v = self._set_multi_index(df_v, f'{sensor}_{loc}')
            df = pd.concat([df, df_v])
        return df

    def _set_multi_index(self, df, sensor):
        df.index = pd.MultiIndex.from_product([[self.subject], df.index])
        df.columns = pd.MultiIndex.from_product([[sensor], df.columns])
        return df

    def _get_location_files(self):
        location_files = []
        if self.annotators is not None:
            for ai, annotator in enumerate(self.annotators):
                filename = f"{self.path}/location_{ai}.csv"
                if self._file_exists(filename):
                    location_files.append(filename)
        return location_files

    @staticmethod
    def _file_exists(filename):
        # checks whether subject has meta, TODO: Adapt to general
        return os.path.isfile(filename)

    @staticmethod
    def _read_annotation(filepath):
        df = pd.read_csv(filepath, na_filter=True)
        df.start = pd.TimedeltaIndex(df.start, unit="s", name="start")
        df.end = pd.TimedeltaIndex(df.end, unit="s", name="end")
        return df

    @staticmethod
    def _read_targets(filepath):
        #     Not needed
        df = pd.read_csv(filepath, na_filter=True)
        df.start = pd.TimedeltaIndex(df.start, unit="s", name="start")
        df.end = pd.TimedeltaIndex(df.end, unit="s", name="end")
        return df

    def load(self):
        self.metadata, self.start, self.end, self.annotators =\
            self._get_meta()
        self._index = self._get_index()

    def read_location(self):
        loc_files = self._get_location_files()
        rooms = self.datasubset.dataset.metadata['rooms']
        df = None
        if loc_files:
            df = pd.DataFrame([])
            i_f = 0
            for i_f, filename in enumerate(loc_files):
                df_loc = pd.read_csv(filename, na_filter=True)
                df_loc.start = pd.TimedeltaIndex(df_loc.start, unit="s", name="start")
                df_loc.end = pd.TimedeltaIndex(df_loc.end, unit="s", name="end")
                if df.empty:
                    df = self._read_by_interval(df_loc, columns=rooms)
                else:
                    df = df + self._read_by_interval(df_loc, columns=rooms)

            df = df / (i_f + 1)
            df = self._set_multi_index(df, "Locations")
        return df

    def read_data(self, **kwargs):
        resample = kwargs.pop("resample", True)
        unit = kwargs.pop("unit", '50L')

        # read csv subjects file per sensor
        data = pd.DataFrame([])
        raw_data = {}
        for sensor in self.datasubset.dataset.sensors:
            exec(f"raw_data['{sensor}']" +
                 f"= self._read_{sensor}()")
            data = pd.concat([data, raw_data[sensor]])
        data.sort_index(inplace=True)

        if resample:
            data = self.resample(data, unit=unit)
            self._index = data.index.droplevel(0)

        return data

    def resample(self, df, **kwargs):
        unit = kwargs.pop("unit", '50L')
        df = df.resample(unit, level=1).mean()
        df.index = pd.MultiIndex.from_product([[self.subject], df.index])
        return df


if __name__ == "__main__":
    print("Hello")
    # import os
