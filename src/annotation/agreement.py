import pandas as pd
import numpy as np

import statsmodels
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa
from sklearn.metrics import cohen_kappa_score


def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)

# # write function for cohens kappa score for 3 annotators
# def cohens_kappa_score(annotator_1_pref, annotator_2_pref, annotator_3_pref):
#     # calculate cohens kappa score
#     kappa = (cohen_kappa(annotator_1_pref, annotator_2_pref) + cohen_kappa(annotator_1_pref, annotator_3_pref) + cohen_kappa(annotator_2_pref, annotator_3_pref))/3
#     return kappa
#
#
# def fleiss_kappa(M):
#     """Computes Fleiss' kappa for group of annotators.
#     :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
#         'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
#     :type: numpy matrix
#     :rtype: float
#     :return: Fleiss' kappa score
#     """
#     N, k = M.shape  # N is # of items, k is # of categories
#     n_annotators = float(np.sum(M[0, :]))  # # of annotators
#     tot_annotations = N * n_annotators  # the total # of annotations
#     category_sum = np.sum(M, axis=0)  # the sum of each category over all items
#
#     # chance agreement
#     p = category_sum / tot_annotations  # the distribution of each category over all annotations
#     PbarE = np.sum(p * p)  # average chance agreement over all categories
#
#     # observed agreement
#     P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
#     Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items
#
#     return round((Pbar - PbarE) / (1 - PbarE), 4)


def checkInput(rate, n):
    """
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer"
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"


def fleissKappa(rate, n):
    """
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    # checkInput(rate, n)

    # mean of the extent to which raters agree for the ith subject
    PA = sum([(sum([i ** 2 for i in row]) - n) / (n * (n - 1)) for row in rate]) / N
    print("PA = ", PA)

    for i in range(k):
        print("Category ", i, " - ", sum([row[i] for row in rate]))

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j ** 2 for j in [sum([rows[i] for rows in rate]) / (N * n) for i in range(k)]])
    print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa


if __name__ == '__main__':
    file = "data/ChatGPT_Prompt_Exploration - English.csv"
    df = pd.read_csv(file)
    print(df.head())

    annotator_1_pref = list(df["H1 Correct"].values)
    annotator_2_pref = list(df["H2 Correct"].values)
    annotator_3_pref = list(df["H3 Correct"].values)
    # replace value of 2 in list with 0
    annotator_1_pref = [0 if x == 2 else x for x in annotator_1_pref]
    annotator_2_pref = [0 if x == 2 else x for x in annotator_2_pref]
    annotator_3_pref = [0 if x == 2 else x for x in annotator_3_pref]
    # count 0 and 1 in list and create a list of list
    # annotator_1_pref = [annotator_1_pref.count(0), annotator_1_pref.count(1)]
    # annotator_2_pref = [annotator_2_pref.count(0), annotator_2_pref.count(1)]
    # annotator_3_pref = [annotator_3_pref.count(0), annotator_3_pref.count(1)]
    # # create a matrix of shape (3, 2) with 3 annotators and 2 categories
    # M = np.array([annotator_1_pref, annotator_2_pref, annotator_3_pref])
    # print(M)
    # score = fleissKappa(M, 3)
    # print(score)
    print(annotator_1_pref)
    print(annotator_2_pref)
    print(annotator_3_pref)

    # convert values in annotator_1_pref to  int
    annotator_1_pref = [int(i) for i in annotator_1_pref]
    annotator_2_pref = [int(i) for i in annotator_2_pref]
    annotator_3_pref = [int(i) for i in annotator_3_pref]

    # # make annotator_1_pref, annotator_2_pref into a list of list
    # for val in annotator_1_pref:
    #     print(type(val))
    #
    # x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    # y = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    # # calculate cohens kappa score
    # kappa12 = cohen_kappa_score(x, y)
    # print(kappa12)
    #
    # rater1 = [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    # rater2 = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
    # kappa12 = cohen_kappa_score(rater1, rater2)
    # print(kappa12)
    # kappa13 = cohens_kappa(annotator_1_pref, annotator_3_pref)
    # print(kappa13)

    # create a matrix for 3 annotators with 2 categories and len list samples
    M = np.zeros((len(annotator_1_pref), 2))
    # print(M)
    # iterate over list and count 0 and 1 and add to matrix
    for i in range(len(annotator_1_pref)):
        M[i][0] = int([annotator_1_pref[i]].count(0) + [annotator_2_pref[i]].count(0) + [annotator_3_pref[i]].count(0))
        M[i][1] = int([annotator_1_pref[i]].count(1) + [annotator_2_pref[i]].count(1) + [annotator_3_pref[i]].count(1))
    print(M)
    #
    # kappa = fleissKappa(M, 3)
    # print(kappa)

    # M = [[1, 1],  # all three raters disagree
    #  [1, 1],  # again
    #  [1, 1],  # and again
    #  [1, 1],  # and again
    #  [0, 2],  # all three raters agree on 1
    #  [1, 1],
    #  [2, 0],  # two raters agree on 0, third picked 3
    #  [2, 0],  # perfect disagreement
    #  [2, 0],  # for the rest of the dataset.
    #  [2, 0]]

    # M = [
    #      [0, 2],
    #      [1, 1],
    #      [0, 2],
    #      [0, 2],
    #      [0, 2],
    #      [0, 2],
    #      [0, 2],
    #      [0, 2]
    #      ]

    # from statsmodels.stats import inter_rater as irr
    #
    # agg = irr.aggregate_raters(M, n_cat=2)
    # print(agg)
    #
    # score = irr.fleiss_kappa(agg[0], method='fleiss')
    # print(score)

    # import numpy as np
    # import krippendorff as kd
    #
    # arrT = np.array(M).transpose()  # returns a list of three lists, one per rater
    # k_score = kd.alpha(arrT, level_of_measurement='nominal')  # assuming nominal categories
    # print(k_score)

    data = pd.DataFrame(M, columns=['0', '1'])
    print(data.head())

    raters = 3
    categories = 2
    subjects = 60

    sum_cells = subjects * raters

    total_m = data['0'].sum()
    total_f = data['1'].sum()
    # total_u = data['Unsure'].sum()

    # print(total_m, total_f, total_u)

    # proportion of assignments that were made to the category of male, female and unsure
    assign_male = round((total_m / sum_cells), 3)
    assign_female = round((total_f / sum_cells), 3)
    # assign_unsure = round((total_u / sum_cells), 3)

    # print(assign_male,assign_female,assign_unsure)

    mean_assign = round((assign_male ** 2) + (assign_female ** 2), 3) #+ (assign_unsure ** 2), 3)

    # print(mean_assign)

    # degree of agreement that is attainable above chance
    deg_agreement = 1 - mean_assign
    print(deg_agreement)


    # calculating the extent to which the raters agree

    def cal_prob(row):
        x = ((row['0'] ** 2) + (row['1'] ** 2) - raters)
        prob = x / (raters * (raters - 1))
        return round(prob, 3)

    data['Probability'] = data.apply(cal_prob, axis=1)
    print(data.head())

    # calculate the mean aggrement for measuring kappa
    prob_mean = round(data['Probability'].sum() / subjects, 3)

    # degree of agreement actually achieved above chance (prob_mean - mean_assign )
    agree_abv_chance = round((prob_mean - mean_assign), 3)
    print(agree_abv_chance)

    # kappa measure
    kappa_measure = round((agree_abv_chance / deg_agreement), 3)
    print(kappa_measure)
