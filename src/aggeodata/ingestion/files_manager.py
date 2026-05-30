"""
aggeodata.ingestion.files_manager
===================================

File and date utilities shared across downloaders.
"""

from __future__ import annotations

import re
from dateutil.parser import parse
from calendar import monthrange
import glob
import os
import zipfile
import numpy as np
from typing import Tuple, Dict, List
import itertools

def split_date(date: str) -> Tuple[int, int, int]:
    """
    Splits a date string in 'YYYY-MM-DD' format into a tuple of integers.

    Parameters
    ----------
    date : str
        Date string in the format 'YYYY-MM-DD'.

    Returns
    -------
    Tuple[int, int, int]
        A tuple containing (year, month, day) as integers.

    Example
    -------
    >>> split_date("2024-10-04")
    (2024, 10, 4)
    """

    if '-' in date:
        year, month, day = tuple(map(int, date.split('-')))
    
    assert month <= 12, f"There are only 12 months month is bigger {month}"

    return [year, month, day]

def months_range_asstring(month_init: int, month_end: int):

    if month_end > 31:
        month_end = 31
        raise Warning('maximun months in a year is 12')
    if month_init < 1:
        month_init = 1
        raise Warning('months must be a positive number')
    
    if month_init != month_end:
        months = [f"{i}" if i > 9 else f"0{i}" for i in range(month_init, month_end+1)]
    else: 
        months = [f"{month_init}" if month_init > 9 else f"0{month_init}"]

    return  months  


def days_range_asstring(day_init: int, day_end: int):

    if day_end > 31:
        day_end = 31
        raise Warning('maximun days in a month is 31')
    if day_init < 1:
        day_init = 1
        raise Warning('days must be a positive number')
    
    days = [f"{i}" if i > 9 else f"0{i}" for i in range(day_init, day_end+1)]

    return  days  


def set_months_and_days(year, init_month, end_month, init_day = None, end_day = None):
        months = months_range_asstring(init_month, end_month)
        month_dict = {}
        for month in months:
            if int(month) == end_month:
                end_dayc = monthrange(year, int(month))[1] if end_day is None else end_day
            else:
                end_dayc = monthrange(year, int(month))[1]

            if int(month) == init_month:
                init_dayc = 1 if init_day is None else init_day
            else:
                init_dayc = 1

            days = days_range_asstring(init_dayc, end_dayc)
            month_dict[month] = days
        return month_dict

def create_yearly_query(init_date, end_date):
    sty, stm, std = split_date(init_date)
    eny, enm, end = split_date(end_date)
    diffyears = eny - sty
    queryyearlydates = {}
    if diffyears != 0:
        for year in range(sty, eny+1):
            
            if year == sty:
                
                month_days = set_months_and_days(year=year, init_day=std, init_month=stm, end_month=12)
            elif year == eny:
                
                month_days = set_months_and_days(year=year, init_month=1, end_month=enm, end_day=end)
            else:
                month_days = set_months_and_days(year=year, init_month=1, end_month=12, init_day = None)

            queryyearlydates[str(year)] =  {i: month_days[i] for i in month_days.keys()}
    else:
        month_days = set_months_and_days(year=sty, init_day=std, init_month=stm, end_month=enm, end_day=end)
        queryyearlydates[str(sty)] =  {i: month_days[i] for i in month_days.keys()}
        
    return queryyearlydates

def uncompress_zip_path(path, year):
    foldermanager = check_filesinzipfolder(glob.glob(path+'/*{}*'.format(year)))
    if foldermanager['unzip']:
        if not os.path.exists(foldermanager['tempfolder']):
            with zipfile.ZipFile(foldermanager['inputfolder'], 'r') as zip_ref:
                zip_ref.extractall(foldermanager['tempfolder'])
        info_path = foldermanager['tempfolder']
    else:
        info_path = foldermanager['inputfolder']

    return info_path


class IntervalFolderManager:
    """
    A class to manage folder operations, split date ranges, and handle zip files in a given directory.

    Attributes
    ----------
    path : str
        Path to the directory containing the files.
    starting_date : str
        Start date of the query in 'YYYY-MM-DD' format.
    ending_date : str
        End date of the query in 'YYYY-MM-DD' format.
    _folders_to_remove : List[str]
        List of folders to be removed after processing.
    _query_dates : dict
        Cached dictionary of dates split into yearly queries.
    
    Methods
    -------
    split_date(date: str) -> Tuple[int, int, int]:
        Split a date string into year, month, and day.
    query_dates() -> dict:
        Generate and cache yearly query dates for the date range.
    range_years() -> List[int]:
        Get a list of years between the start and end dates.
    checkzip_folders(year: str, extension: str = '.zip') -> str:
        Extract files from zip folders and return the file path.
    check_path() -> None:
        Assert that the specified path exists.
    split_dates_for_interval() -> None:
        Split the start and end dates into year, month, and day components.
    __call__(path: str, starting_date: str, ending_date: str) -> List[List[str]]:
        Process files for each year in the date range and return a list of matching file names.
    """

    @staticmethod
    def split_date(date):
        return split_date(date)
    @property
    def query_dates(self):
        if self._query_dates is None:
            self._query_dates = create_yearly_query(init_date=self.starting_date, end_date=self.ending_date)
        
        return self._query_dates
    
    def range_years(self):

        return list(range(self._ys,self._ye+1))


    def __init__(self, ) -> None:


        self._folders_to_remove = []
        self._query_dates = None
        self.path = ""
        self.starting_date = ""
        self.ending_date = ""
        self._ys = self._ms = self._ds = 0
        self._ye = self._me = self._de = 0
    
    def check_and_extract_zip(self, year, extension = '.zip'):
        """
        Checks if the folder contains a zip file and extracts its contents if needed.

        Parameters
        ----------
        year : str
            Year to filter files.
        extension : str, optional
            File extension (default is '.zip').

        Returns
        -------
        str
            Path to the folder containing the extracted or original files.
        """
        if extension == ".zip":
            return uncompress_zip_path(self.path, year)
        

    def check_path_exists(self):
        assert os.path.exists(self.path), "The path does not exist"

    def split_dates(self):
        
        self._ys, self._ms, self._ds = self.split_date(self.starting_date)
        self._ye, self._me, self._de = self.split_date(self.ending_date)

    def initialize(self, path: str, starting_date: str, ending_date: str) -> None:
        """
        Sets the path, start, and end dates for the instance, splitting the date components.

        Parameters
        ----------
        path : str
            Path to the folder containing the files.
        starting_date : str
            Start date in 'YYYY-MM-DD' format.
        ending_date : str
            End date in 'YYYY-MM-DD' format.
        """
        self.path = path
        self.check_path_exists()
        self.starting_date = starting_date
        self.ending_date = ending_date
        self.split_dates()

    def __call__(self, path: str, starting_date: str, ending_date: str) -> List[List[str]]:
        """
        Process files for each year in the date range and return a list of matching file names.

        Parameters
        ----------
        path : str
            Path to the directory containing the files.
        starting_date : str
            Start date in 'YYYY-MM-DD' format.
        ending_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        List[List[str]]
            A list of lists where each sublist contains the date and file path for files matching the query.
        """

        self.initialize(path, starting_date, ending_date)

        listfilesyear = []
        for year in self.range_years():
            path = self.check_and_extract_zip('{}'.format(year))
            print(path)
            dates_toquery = concatenate_dates(str(year), self.query_dates)
            _VALID_EXTS = ('.nc', '.tif', '.tiff', '.zip')
            file_names = [
                [d, filepath]
                for filepath in os.listdir(path)
                for d in dates_toquery
                if filepath.find(d) != -1 and filepath.lower().endswith(_VALID_EXTS)
            ]
            listfilesyear.append(file_names)

        return list(itertools.chain.from_iterable(listfilesyear))
    
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def find_date_instring(string, pattern = "202", yearformat = 'yyyy'):
    """find date pattern in a string

    Args:
        string (_type_): string
        pattern (str, optional): date init. Defaults to "202".
        yearformat (str, optional): the year format in the string 2021 is yyyy. Defaults to 'yyyy'.

    Returns:
        string: date in yyyymmdd format
    """
    matches = re.finditer(pattern, string)
    
    if yearformat == 'yyyy':
        datelen = 8
    else:
        datelen = 6
    
    matches_positions = [string[match.start():match.start() +datelen] 
                                for match in matches if is_date(string[match.start():match.start() +datelen])]
    if len(matches_positions[0]) == 6:
        matches_positions = [pattern[:-1]+matches_positions[0]]
    
    return matches_positions[0]


def check_filesinzipfolder(folder):
    folder = folder if isinstance(folder, list) else [folder]
    zipfolder = [i for i in folder if i.endswith('.zip')]
    out = {}
    if len(zipfolder) == 1:
        out['inputfolder'] = zipfolder[0]
        out['tempfolder'] = zipfolder[0][0:zipfolder[0].index('.zip')]
        out['unzip'] = True
    else:
        out['inputfolder'] = folder[0]
        out['unzip'] = False
    return out


def concatenate_dates(year, dict_dates,sep = ''):
    cdates = []
    for month in dict_dates[year].keys():
        for day in dict_dates[year][month]:
            cdates.append('{}{}{}{}{}'.format(year,sep, month, sep, day))
    return cdates



class SoilFolderManager:
    @staticmethod
    def extract_detph(paths, variable, units = 'cm'):
        depths = []
        for path in paths:
    
            matches = list(re.finditer(variable, path))[-1]
            depths.append(path[matches.end()+1:path.index(units+'_')])

        return depths
    @staticmethod
    def _sort_depths(depths):
        init_depths = [i.split('-')[0] for i in depths]
        sortindex =  list(np.argsort(np.array(init_depths).astype(int)))
        depths = [str(depths[i]) for i in sortindex]
        return depths, sortindex
        

    def get_all_paths(self, units_string = 'cm', by = 'depth'):
        paths_dict = {}

        for var in self.variables:
            varinfo = self.variable_path(var, units_string = units_string)
            if varinfo is not None:
                paths_dict[var] = varinfo
        
        depths_available = [self.extract_detph(v, k, units = units_string) for k,v in paths_dict.items()]
        self.depths = self._sort_depths(np.unique(depths_available))[0]
        
        if by == 'depth':
            paths_dict_ = {}
            for i, depth in enumerate(self.depths):
                varpaths = {}
                for k,v in paths_dict.items():
                    if i <= len(v):
                        varpaths[k] = v[i]
                paths_dict_[depth] = varpaths
            paths_dict = paths_dict_
        return paths_dict

    def variable_path(self, variable, units_string = 'cm'):
        
        variable_paths = self._check_variable_paths(variable)
        if len(variable_paths) == 0:
            print(f'there is not data for this variable {variable}')
            return None
        
        self._extract_depths(variable, units_string = units_string)

        return [variable_paths[i] for i in self._depthssorted]

    def _extract_depths(self, variable, units_string = 'cm'):
        variable_paths = self._check_variable_paths(variable)
        if len(variable_paths)>0:
            depths = self.extract_detph(variable_paths, variable, units = units_string)
            self.depths, self._depthssorted = self._sort_depths(depths)
        else:
            self._depthssorted = list(range(len(variable_paths)))

        return self.depths

    def _check_variable_paths(self, variable):
        return glob.glob(self.path+'/*{}*{}'.format(variable, self._extension))

    def __init__(self, path, variables, raster_extension = '.tif') -> None:
        self.depths = None
        self.path = path
        self.variables = variables
        self._extension = raster_extension