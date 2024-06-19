from strsimpy import *
import pandas as pd
from difflib import SequenceMatcher
import math

def similar_normalized_levenshtein(str_1: str = '', str_2: str = ''):
    """
    The function calculates the normalized Levenshtein distance between two input strings.
    
    :param str_1: The function `similar_normalized_levenshtein` calculates the normalized Levenshtein
    distance between two input strings `str_1` and `str_2`. The Levenshtein distance is a measure of the
    similarity between two strings, representing the minimum number of single-character edits (insert
    :type str_1: str
    :param str_2: It seems like you have defined a function `similar_normalized_levenshtein` that
    calculates the normalized Levenshtein distance between two strings `str_1` and `str_2`. The function
    returns the similarity score as a float
    :type str_2: str
    :return: The function `similar_normalized_levenshtein` returns a floating-point value representing
    the similarity score between the two input strings `str_1` and `str_2` calculated using the
    Normalized Levenshtein distance algorithm.
    """
    similar_score = normalized_levenshtein.NormalizedLevenshtein().distance(str_1, str_2)
    return float(similar_score)

def similar_metric_lcs(str_1: str = '', str_2: str = ''):
    """
    The function `similar_metric_lcs` calculates the similarity score between two strings using the
    MetricLCS algorithm.
    
    :param str_1: It looks like you are trying to calculate the similarity score between two strings
    using the MetricLCS algorithm. However, there seems to be a mistake in your code
    :type str_1: str
    :param str_2: It looks like you are trying to calculate the similarity score between two strings
    using the MetricLCS algorithm. However, there seems to be a mistake in your code
    :type str_2: str
    :return: The function is attempting to return the result of the `similar_metric_lcs` function, which
    is the similarity score calculated using the MetricLCS algorithm between the two input strings
    `str_1` and `str_2`. However, there is a mistake in the code where the return statement is trying to
    return a variable `similar_metric_lcs` which is not defined.
    """
    similare_score = metric_lcs.MetricLCS().distance(str_1, str_2)
    return float(similare_score)

def similar_ngram_4(str_1: str = '', str_2: str = ''):
    """
    The function `similar_ngram_4` calculates the similarity score between two strings based on their
    4-gram distance.
    
    :param str_1: The `similar_ngram_4` function you provided calculates the similarity score between
    two strings using the N-Gram distance metric with N=4. This metric measures the similarity between
    two strings by counting the number of common sequences of N items (in this case, N=4) between them
    :type str_1: str
    :param str_2: Thank you for providing the code snippet. It looks like you have defined a function
    `similar_ngram_4` that calculates the similarity score between two strings using 4-gram distance
    :type str_2: str
    :return: The function `similar_ngram_4` returns a floating-point value representing the similarity
    score between two input strings `str_1` and `str_2` calculated using the N-Gram distance metric with
    N=4.
    """
    similar_score = ngram.NGram(4).distance(str_1, str_2)
    return float(similar_score)

def similar_cosine(str_1: str = '', str_2: str = ''):
    """
    The function `similar_cosine` calculates the cosine similarity score between two input strings.
    
    :param str_1: The `similar_cosine` function takes two strings `str_1` and `str_2`, calculates their
    cosine similarity score using a library called `cosine`, and returns the similarity score as a float
    :type str_1: str
    :param str_2: It seems like you have provided a code snippet for calculating the cosine similarity
    between two strings using a library called `cosine`. The function `similar_cosine` takes two string
    inputs `str_1` and `str_2`, calculates their cosine similarity score, and returns the result as a
    float
    :type str_2: str
    :return: The function `similar_cosine` returns a float value representing the similarity score
    between two input strings `str_1` and `str_2` calculated using cosine similarity.
    """
    p1 = cosine.Cosine(4).get_profile(str_1)
    p2 = cosine.Cosine(4).get_profile(str_2)
    similar_score = cosine.Cosine(4).similarity_profiles(p1, p2)
    return float(similar_score)

def benmarking_metric_similar(str_1: str = '', str_2: str = '', list_conf: list = []):
    """
    The function `benmarking_metric_similar` calculates various similarity metrics between two strings
    and appends the results to a given list.
    
    :param str_1: The function `benmarking_metric_similar` takes three parameters: `str_1`, `str_2`, and
    `list_conf`. It calculates and appends the results of four similarity metrics (normalized
    Levenshtein, LCS, n-gram, and cosine similarity) between `str
    :type str_1: str
    :param str_2: It looks like you have defined a function `benmarking_metric_similar` that takes in
    two strings `str_1` and `str_2`, and a list `list_conf`. The function then appends the results of
    four similarity metrics (similar_normalized_levenshtein, similar
    :type str_2: str
    :param list_conf: The `list_conf` parameter is a list that stores the results of different
    similarity metrics calculated between two strings `str_1` and `str_2`. The metrics being calculated
    and appended to the list are:
    :type list_conf: list
    :return: The function `benmarking_metric_similar` is returning a list `list_conf` that contains the
    results of four similarity metrics calculated between the input strings `str_1` and `str_2`. The
    similarity metrics being calculated and appended to the list are:
    1. Similarity using normalized Levenshtein distance
    2. Similarity using Longest Common Subsequence (LCS)
    """

    list_conf.append([similar_normalized_levenshtein(str_1 = str_1, str_2= str_1),
                      similar_metric_lcs(str_1= str_1, str_2= str_2),
                      similar_ngram_4(str_1= str_1, str_2= str_2),
                      similar_cosine(str_1= str_1, str_2= str_2)])
    return list_conf

# def select_feature_from_dataframe(df_metric: pd.DataFrame = ''):
#     min_levenshtein_idx = df_metric['levenshtein'].idxmin()
#     min_lcs_idx = df_metric['lcs'].idxmin()
#     min_ngram_idx = df_metric['ngram'].idxmin()

#     max_cosine_idx = df_metric['cosine'].idxmax()
#     list_max_cosine = []
#     max_current = df_metric['cosine'][max_cosine_idx]
#     for idx_cosine in range(max_cosine_idx, len(df_metric['cosine'])):
#         if df_metric['cosine'][idx_cosine] == max_current:
#             list_max_cosine.append(idx_cosine)

#     min_leven = 1
#     min_lcs = 1
#     min_ngram = 1

#     feature_select_leve = 0
#     fea

#     for idx_remain in list_max_cosine:
#         current_leven = df_metric['levenshtein'][idx_remain] - df_metric['levenshtein'][min_levenshtein_idx]
#         current_lcs = df_metric['lcs'][idx_remain] - df_metric['lcs'][min_lcs_idx]
#         current_ngram = df_metric['ngram'][idx_remain] - df_metric['ngram'][min_ngram_idx]
#         if 

def calculator_similarity_str(str_1: str = "", str_2: str = ""):
    return SequenceMatcher(None, str_1, str_2).ratio()

def select_feature_from_dataframe(df_metric: pd.DataFrame = ''):
    max_cosine_idx = df_metric['cosine'].idxmax()
    list_max_cosine = []
    max_current = df_metric['cosine'][max_cosine_idx]
    for idx_cosine in range(max_cosine_idx, len(df_metric['cosine'])):
        if df_metric['cosine'][idx_cosine] == max_current:
            list_max_cosine.append(idx_cosine)

    minium_gram = 1
    id_minium_gram = None
    for i in list_max_cosine:
        if df_metric['ngram'][i] < minium_gram:
            minium_gram = df_metric['ngram'][i]
            id_minium_gram = i
    
    return id_minium_gram


def distance_to_bbox(point, bbox):
    px, py = point
    xmin, ymin, xmax, ymax = bbox
    
    if px < xmin:
        dx = xmin - px
    elif px > xmax:
        dx = px - xmax
    else:
        dx = 0
    
    if py < ymin:
        dy = ymin - py
    elif py > ymax:
        dy = py - ymax
    else:
        dy = 0
    
    return math.sqrt(dx*dx + dy*dy)

def find_nearest_and_second_nearest_bbox(point, bbox_list):
    if not bbox_list or len(bbox_list) < 2:
        return None, None
    
    min_distance = float('inf')
    second_min_distance = float('inf')
    nearest_bbox = None
    second_nearest_bbox = None
    
    for bbox in bbox_list:
        distance = distance_to_bbox(point, bbox)
        
        if distance < min_distance:
            second_min_distance = min_distance
            min_distance = distance
            second_nearest_bbox = nearest_bbox
            nearest_bbox = bbox
        elif distance < second_min_distance:
            second_min_distance = distance
            second_nearest_bbox = bbox
    
    return nearest_bbox, second_nearest_bbox

