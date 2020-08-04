import os
from pathlib import Path
from sklearn.utils import Bunch
from nilearn.datasets.utils import _get_dataset_dir, _fetch_files
from nilearn._utils.numpy_conversions import csv_to_array


def _fetch_hbnssi_participants(data_dir, url, verbose):
    """
    Helper function to fetch_hbnssi.
    This function helps in downloading and loading participants data from .tsv
    uploaded on Open Science Framework(OSF).

    Parameters
    ----------
    data_dir: str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.
    url: str, optional
        Override download URL. Used for test only(or if you setup a mirror of
        the data). Default: None
    verbose: int
        Defines the level of verbosity of the output.

    Returns
    -------
    participants: numpy.ndarray
        Contains data of each subject age, gender, handedness.
    """
    if url is None:
        url = 'https://osf.io/wtvh3/download'

    files = [('participants.csv', url, {'move': 'participants.csv'})]
    path_to_participants = _fetch_files(data_dir, files, verbose=verbose)[0]

    # Load path to participants
    dtype = [('sid', 'U12'), ('age', '<f8'), ('Gender', 'U4'),
             ('Handedness', 'U4')]
    names = ['sid', 'age', 'gender', 'handedness']
    participants = csv_to_array(path_to_participants, skip_header=True,
                                dtype=dtype, names=names)
    return participants


def _fetch_hbnssi_brain_mask(data_dir, url, verbose):
    """
    Helper function to fetch_hbnssi.
    This function helps in downloading and loading the brain mask
    from Open Science Framework(OSF).

    Parameters
    ----------
    data_dir: str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.
    url: str, optional
        Override download URL. Used for test only(or if you setup a mirror of
        the data). Default: None
    verbose: int
        Defines the level of verbosity of the output.

    Returns
    -------
    path_to_mask: str
        File path for the appropriate brain mask
    """
    if url is None:
        url = 'https://osf.io/kp6m9/download'

    target_fname = 'tpl-MNI152NLin2009cAsym_res-3mm_label-GM_desc-thr02_probseg.nii.gz'
    files = [(target_fname,
             url,
             {'move': target_fname})]
    path_to_mask = _fetch_files(data_dir, files, verbose=verbose)[0]

    return path_to_mask


def _fetch_hbnssi_functional(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_development_fmri.

    This function helps in downloading functional MRI data in Nifti
    and its confound corresponding to each subject.

    The files are downloaded from Open Science Framework (OSF).

    Parameters
    ----------
    participants : numpy.ndarray
        Should contain column participant_id which represents subjects id. The
        number of files are fetched based on ids in this column.

    data_dir: str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.

    url: str, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data). Default: None

    resume: bool, optional (default True)
        Whether to resume download of a partly-downloaded file.

    verbose: int
        Defines the level of verbosity of the output.

    Returns
    -------
    func: list of str (Nifti files)
        Paths to functional MRI data (4D) for each subject.
    """
    if url is None:
        # Download from the relevant OSF project, using hashes generated
        # from the OSF API. Note the trailing slash. For more info, see:
        # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
        url = 'https://osf.io/download/{}/'

    raiders = 'sub-{0}_task-RAIDERS_space-MNI152NLin2009cAsym_desc-postproc_bold.nii.gz'
    flanker = 'sub-{0}_task-FLANKERTASK_space-MNI152NLin2009cAsym_desc-postproc_bold.nii.gz'
    conditions = 'sub-{0}_labels.csv'
    runs = 'sub-{0}_runs.csv'

    # The gzip contains unique download keys per Nifti file and confound
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('sid', 'U12'), ('raiders', 'U24'), ('flanker', 'U24'),
             ('condition', 'U24'), ('run', 'U24')]
    names = ['sid', 'raiders', 'flanker', 'condition', 'run']
    # csv file contains download information related to OpenScience(osf)
    osf_data = csv_to_array(os.path.join(package_directory, "hbnssi.csv"),
                            skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'derivatives')
    align, decode, labels, sessions = [], [], [], []

    for sid in participants['sid']:
        this_osf_id = osf_data[osf_data['sid'] == sid]

        # Download raiders
        raiders_url = url.format(this_osf_id['raiders'][0])
        raiders_target = Path(derivatives_dir, raiders.format(sid))
        raiders_file = [(raiders_target,
                         raiders_url,
                         {'move': raiders_target})]
        path_to_raiders = _fetch_files(data_dir, raiders_file,
                                       verbose=verbose)[0]
        align.append(path_to_raiders)

        # Download flanker
        flanker_url = url.format(this_osf_id['flanker'][0])
        flanker_target = Path(derivatives_dir, flanker.format(sid))
        flanker_file = [(flanker_target,
                         flanker_url,
                         {'move': flanker_target})]
        path_to_flanker = _fetch_files(data_dir, flanker_file,
                                       verbose=verbose)[0]
        decode.append(path_to_flanker)

        # Download condition labels
        label_url = url.format(this_osf_id['condition'][0])
        label_target = Path(derivatives_dir, conditions.format(sid))
        label_file = [(label_target,
                      label_url,
                      {'move': label_target})]
        path_to_labels = _fetch_files(data_dir, label_file,
                                      verbose=verbose)[0]
        labels.append(path_to_labels)

        # Download session run numbers
        session_url = url.format(this_osf_id['run'][0])
        session_target = Path(derivatives_dir, runs.format(sid))
        session_file = [(session_target,
                        session_url,
                        {'move': session_target})]
        path_to_sessions = _fetch_files(data_dir, session_file,
                                        verbose=verbose)[0]
        sessions.append(path_to_sessions)

    return derivatives_dir


def fetch_hbnssi(data_dir=None, resume=True, verbose=1):
    """Fetch the Healthy Brain Network - Serial Scanning Initative data.

    The data is downsampled to 3mm isotropic resolution. Please see
    Notes below for more information on the dataset as well as full
    pre- and post-processing details.


    Parameters
    ----------
    data_dir: str, optional (default None)
        Path of the data directory. Used to force data storage in a specified
        location. If None, data are stored in home directory.
    resume: bool, optional (default True)
        Whether to resume download of a partly-downloaded file.
    verbose: int, optional (default 1)
        Defines the level of verbosity of the output.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :

        - 'func': list of str (Nifti files)
            Paths to downsampled functional MRI data (4D) for each subject.

        - 'phenotypic': numpy.ndarray
            Contains each subject age, age group, child or adult, gender,
            handedness.

    Notes
    -----
    The original data is downloaded from the HBN-SSI data portal:
    http://fcon_1000.projects.nitrc.org/indi/hbn_ssi/

    This fetcher downloads downsampled data that are available on Open
    Science Framework (OSF): https://osf.io/28qwv/files/

    Pre- and post-processing details for this dataset are made available
    here: https://osf.io/28qwv/wiki/

    References
    ----------
    Please cite this paper if you are using this dataset:

    O'Connor D, Potler NV, Kovacs M, Xu T, Ai L, Pellman J, Vanderwal T,
    Parra LC, Cohen S, Ghosh S, Escalera J, Grant-Villegas N, Osman Y, Bui A,
    Craddock RC, Milham MP (2017). The Healthy Brain Network Serial Scanning
    Initiative: a resource for evaluating inter-individual differences and
    their reliabilities across scan conditions and sessions.
    GigaScience, 6(2): 1-14
    https://academic.oup.com/gigascience/article/6/2/giw011/2865212
    """

    dataset_name = "hbnssi"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=1)

    # Participants data
    participants = _fetch_hbnssi_participants(data_dir=data_dir, url=None,
                                              verbose=verbose)
    brain_mask = _fetch_hbnssi_brain_mask(data_dir=data_dir, url=None,
                                          verbose=verbose)

    derivatives_dir = _fetch_hbnssi_functional(
        participants, data_dir=data_dir, url=None,
        resume=resume, verbose=verbose)

    # create out_dir
    Path(data_dir, "decoding").mkdir(parents=True, exist_ok=True)
    out_dir = Path(data_dir, "decoding")
    # create mask_cache
    Path(data_dir, "mask_cache").mkdir(parents=True, exist_ok=True)
    mask_cache = Path(data_dir, "mask_cache")

    return Bunch(subjects=participants['sid'], mask=brain_mask,
                 task_dir=derivatives_dir, out_dir=out_dir,
                 mask_cache=mask_cache)
