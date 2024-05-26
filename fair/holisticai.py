# https://github.com/holistic-ai/holisticai/tree/main


import pandas as pd
import numpy as np


def _check_nan(arr, name=""):
    """
    Check for nan

    Description
    ----------
    This function checks if a numpy array
    has a nan.

    Parameters
    ----------
    arr : numpy array
        input
    name : str
        the name of the input

    Returns
    -------
    ValueError or None
    """
    if pd.isnull(pd.Series(arr)).any():
        raise ValueError(name + " has NaN values.")


def _check_groups_input(groups, p_attr):
    """
    Check groups input (multiclass)

    Description
    ----------
    This function checks the groups input is
    composed of the unique values of p_attr

    Parameters
    ----------
    groups : numpy array
        Class vector (categorical)
    p_attr : numpy array
        Predictions vector (categorical)

    Returns
    -------
    ValueError or None
    """
    # compare sets
    _groups = set(groups)
    __groups = set(p_attr)
    if _groups != __groups:
        raise ValueError("groups is not a reordering of unique values in p_attr.")


def _check_classes_input(classes, y_pred, y_true=None):
    """
    Check classes input (multiclass)

    Description
    ----------
    This function checks the input classes is
    composed of the unique values of y_pred (and y_true)

    Parameters
    ----------
    classes : numpy array
        Class vector (categorical)
    y_pred : numpy array
        Predictions vector (categorical)
    y_true : numpy array
        Target vector (categorical)

    Returns
    -------
    ValueError or None
    """
    # case 1 : y_true is None
    if y_true is None:
        _classes = set(classes)
        __classes = set(y_pred)
        if _classes != __classes:
            raise ValueError("classes is not a reordering of unique values in y_pred.")
    # case 2
    else:
        _classes = set(classes)
        __classes = set(y_pred).union(set(y_true))
        if _classes != __classes:
            raise ValueError(
                "classes is not a reordering of unique values in y_pred or y_true."
            )


def _check_same_shape(list_of_arr, names=""):
    """
    Check same shape

    Description
    ----------
    This function checks if all numpy arrays
    in a list have same length

    Parameters
    ----------
    list_of_arr : list of numpy arrays
        input
    names : str
        the name of the inputs

    Returns
    -------
    ValueError or None
    """
    num_dims = len(list_of_arr[0].shape)
    for i in range(num_dims):
        try:
            n = len(np.unique([x.shape[i] for x in list_of_arr]))
            if n > 1:
                raise ValueError(names + " do not all have the same shape.")
        except:
            raise ValueError(names + " do not all have the same shape.")


def _array_like_to_numpy(arr, name=""):
    """
    Coerce input to numpy (if possible)

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.

    Parameters
    ----------
    arr : array-like
        Input to coerce

    Returns
    -------
    numpy array or TypeError
    """
    try:
        out = np.squeeze(np.asarray(arr))
        if len(out.shape) == 1:
            return out
        else:
            raise ValueError()
    except:
        raise TypeError(
            "input {} is not array-like. \
This includes numpy 1d arrays, lists, \
pandas Series or pandas 1d DataFrame".format(
                name
            )
        )


def _coerce_and_check_arr(arr, name="input"):
    """
    Coerce and check array-like

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.
    Also checks for nan values.

    Parameters
    ----------
    arr : array-like
        Input to coerce
    name : str
        The name of array

    Returns
    -------
    numpy array or TypeError
    """
    # coerce to numpy if possible
    np_arr = _array_like_to_numpy(arr, name=name)
    # check for nan values
    _check_nan(np_arr, name=name)
    # return
    return np_arr


def _multiclass_checks(
        p_attr=None, y_pred=None, y_true=None, groups=None, classes=None
):
    """
    Multiclass checks

    Description
    ----------
    This function checks inputs to
    a multiclass task

    Returns
    -------
    coerced inputs
    """
    if p_attr is not None:
        p_attr = _coerce_and_check_arr(p_attr, name="p_attr")

    if y_pred is not None:
        y_pred = _coerce_and_check_arr(y_pred, name="y_pred")

    if y_true is not None:
        y_true = _coerce_and_check_arr(y_true, name="y_true")

    # length check
    if p_attr is not None and y_pred is not None:
        _check_same_shape([p_attr, y_pred], names="p_attr, y_pred")

    # length check
    if y_pred is not None and y_true is not None:
        _check_same_shape([y_pred, y_true], names="y_pred, y_true")

    # define groups if not defined (where possible)
    if groups is None:
        if p_attr is not None:
            groups = np.sort(np.unique(p_attr))
        else:
            groups = None
    else:
        _check_groups_input(groups, p_attr)

    # define classes if not defined (where possible)
    if classes is None:
        if y_true is not None and y_pred is not None:
            classes = np.sort(np.unique(np.concatenate((y_pred, y_true))))
        elif y_true is None and y_pred is not None:
            classes = np.sort(np.unique(y_pred))
        else:
            classes = None
    else:
        if y_true is not None and y_pred is not None:
            _check_classes_input(classes, y_pred, y_true)
        elif y_true is None and y_pred is not None:
            _check_classes_input(classes, y_pred)
        else:
            classes = None

    return p_attr, y_pred, y_true, groups, classes


def frequency_matrix(p_attr, y_pred, groups=None, classes=None, normalize="group"):
    """
    Frequency Matrix.

    Description
    ----------
    This function computes the frequency matrix. For each
    group, class pair we compute the count of that group
    for admission within that class. We include the option to normalise
    over groups or classes. By default we normalise by 'group'.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order
    normalize (optional): None, 'group' or 'class'
        According to which of group or class we normalize

    Returns
    -------
    pandas DataFrame
        Success Rate Matrix : shape (num_groups, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> frequency_matrix(p_attr, y_pred, normalize='class')
        0     1     2
    A  0.50  0.25  0.25
    B  0.25  0.50  0.25
    C  0.50  0.00  0.50
    """
    # check and coerce inputs
    p_attr, y_pred, _, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)
    class_dict = dict(zip(classes, range(num_classes)))
    group_dict = dict(zip(groups, range(num_groups)))

    # initialize success rate matrix
    sr_mat = np.zeros((num_groups, num_classes))

    # loop over instances
    for x, y in zip(p_attr, y_pred):
        sr_mat[group_dict[x], class_dict[y]] += 1

    # normalize is None, return counts
    if normalize is None:
        return pd.DataFrame(sr_mat, columns=classes).set_index(np.array(groups))

    # normalise over rows
    elif normalize == "group":
        sr_mat = sr_mat / sr_mat.sum(axis=1).reshape(-1, 1)
        return pd.DataFrame(sr_mat, columns=classes).set_index(np.array(groups))

    # normalise over columns
    elif normalize == "class":
        sr_mat = sr_mat / sr_mat.sum(axis=0).reshape(1, -1)
        return pd.DataFrame(sr_mat, columns=classes).set_index(np.array(groups))

    else:
        raise ValueError("normalize has to be one of None, 'group' or 'class'")


def confusion_tensor(
        p_attr, y_pred, y_true, groups=None, classes=None, as_tensor=False
):
    """
    Confusion Tensor.

    Description
    ----------
    This function computes the confusion tensor. The k,i,jth
    entry is the number of instances of group k with predicted
    class i and true class j.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector (categorical)
    y_pred : array-like
        Prediction vector (categorical)
    y_true : array-like
        Target vector (categorical)
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order
    as_tensor (optional) : bool, default False
        Whether we return a tensor or DataFrame

    Returns
    -------
    numpy ndarray
        Confusion Tensor : shape (num_groups, num_classes, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai import confusion_tensor
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> confusion_tensor(p_attr, y_pred, y_true, as_tensor=True).shape
    (3, 3, 3)
    >>> confusion_tensor(p_attr, y_pred, y_true, as_tensor=True)
    array([[[2., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.]],

        [[0., 0., 1.],
            [0., 2., 0.],
            [1., 0., 0.]],

        [[0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 1.]]])
    >>> confusion_tensor(p_attr, y_pred, y_true, as_tensor=False)
        A	        B	        C
        0	1	2	0	1	2	0	1	2
    0	2.0	0.0	0.0	0.0	0.0	1.0	0.0	1.0	0.0
    1	0.0	1.0	0.0	0.0	2.0	0.0	0.0	0.0	0.0
    2	0.0	1.0	0.0	1.0	0.0	0.0	0.0	0.0	1.0
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)
    class_dict = dict(zip(classes, range(num_classes)))
    group_dict = dict(zip(groups, range(num_groups)))

    # initialize the confusion tensor
    conftens = np.zeros((num_groups, num_classes, num_classes))

    # loop over instances
    for x, y, z in zip(p_attr, y_pred, y_true):
        # increment correct entry
        conftens[group_dict[x], class_dict[y], class_dict[z]] += 1

    # return as a tensor
    if as_tensor is True:
        return conftens

    # return as pandas DataFrame
    elif as_tensor is False:
        d = {}
        for i, group in enumerate(groups):
            # confusion matrix of group number i
            d[group] = pd.DataFrame(conftens[i, :, :], columns=classes).set_index(
                np.array(classes)
            )
        # create a multilevel pandas dataframe
        multi_df = pd.concat(d, axis=1)
        return multi_df

    else:
        raise ValueError("as_tensor should be boolean")


def multiclass_statistical_parity(
        p_attr, y_pred, groups=None, classes=None, aggregation_fun="mean"
):
    """
    Multiclass statistical parity.

    Description
    ----------
    This function computes statistical parity for a classification task
    with multiple classes and a protected attribute with multiple groups.
    For each group compute the vector of success rates for entering
    each class. Compute all distances (mean absolute deviation) between
    such vectors. Then aggregate them using the mean, or max strategy.

    Interpretation
    --------------
    The accepted values and bounds for this metric are the same
    as the 1d case. A value of 0 is desired. Values below 0.1
    are considered fair.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order
    aggregation_fun (optional) : str
        The function to aggregate across groups ('mean' or 'max')

    Returns
    -------
    float
        Multiclass Statistical Parity

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai import multiclass_statistical_parity
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> multiclass_statistical_parity(p_attr, y_pred, aggregation_fun='max')
    0.5
    """
    # check and coerce inputs
    p_attr, y_pred, _, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=groups,
        classes=classes,
    )

    # variables
    num_groups = len(groups)

    # compute frequency matrix (normalised by class)
    sr_mat = frequency_matrix(p_attr, y_pred, groups, classes).to_numpy() + 1e-32

    # initialize distance matrix
    dist_mat = np.zeros((num_groups, num_groups))

    # distance confusion matrix across groups
    for k in range(num_groups):
        pred_prob_k = sr_mat[k]
        for j in range(k + 1, num_groups):
            pred_prob_j = sr_mat[j]
            dist_mat[k, j] = np.abs(pred_prob_k - pred_prob_j).sum() / 2

    if aggregation_fun == "max":
        res = np.max(dist_mat)
    else:
        res = np.sum(dist_mat) / (num_groups * (num_groups - 1) / 2)

    return res


