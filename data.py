import logging
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

partb_aggregated_cols = [
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_Gndr",
    "exclusion",
    "Tot_Srvcs_mean",
    "Tot_Srvcs_median",
    "Tot_Srvcs_sum",
    "Tot_Srvcs_std",
    "Tot_Srvcs_min",
    "Tot_Srvcs_max",
    "Tot_Benes_mean",
    "Tot_Benes_median",
    "Tot_Benes_sum",
    "Tot_Benes_std",
    "Tot_Benes_min",
    "Tot_Benes_max",
    "Tot_Bene_Day_Srvcs_mean",
    "Tot_Bene_Day_Srvcs_median",
    "Tot_Bene_Day_Srvcs_sum",
    "Tot_Bene_Day_Srvcs_std",
    "Tot_Bene_Day_Srvcs_min",
    "Tot_Bene_Day_Srvcs_max",
    "Avg_Sbmtd_Chrg_mean",
    "Avg_Sbmtd_Chrg_median",
    "Avg_Sbmtd_Chrg_sum",
    "Avg_Sbmtd_Chrg_std",
    "Avg_Sbmtd_Chrg_min",
    "Avg_Sbmtd_Chrg_max",
    "Avg_Mdcr_Pymt_Amt_mean",
    "Avg_Mdcr_Pymt_Amt_median",
    "Avg_Mdcr_Pymt_Amt_sum",
    "Avg_Mdcr_Pymt_Amtt_std",
    "Avg_Mdcr_Pymt_Amt_min",
    "Avg_Mdcr_Pymt_Amt_max",
]

partd_aggregated_cols = [
    "Prscrbr_Type",
    "exclusion",
    "Tot_Clms_mean",
    "Tot_Clms_median",
    "Tot_Clms_sum",
    "Tot_Clms_std",
    "Tot_Clms_min",
    "Tot_Clms_max",
    "Tot_30day_Fills_mean",
    "Tot_30day_Fills_median",
    "Tot_30day_Fills_sum",
    "Tot_30day_Fills_std",
    "Tot_30day_Fills_min",
    "Tot_30day_Fills_max",
    "Tot_Day_Suply_mean",
    "Tot_Day_Suply_median",
    "Tot_Day_Suply_sum",
    "Tot_Day_Suply_std",
    "Tot_Day_Suply_min",
    "Tot_Day_Suply_max",
    "Tot_Drug_Cst_mean",
    "Tot_Drug_Cst_median",
    "Tot_Drug_Cst_sum",
    "Tot_Drug_Cst_std",
    "Tot_Drug_Cst_min",
    "Tot_Drug_Cst_max",
    "Tot_Benes_mean",
    "Tot_Benes_median",
    "Tot_Benes_sum",
    "Tot_Benest_std",
    "Tot_Benes_min",
    "Tot_Benest_max",
]

dmepos_aggregated_cols = [
    "Rfrg_Prvdr_Type",
    "Rfrg_Prvdr_Gndr",
    "exclusion",
    "Tot_Suplrs_mean",
    "Tot_Suplrs_median",
    "Tot_Suplrs_sum",
    "Tot_Suplrs_std",
    "Tot_Suplrs_min",
    "Tot_Suplrs_max",
    "Tot_Suplr_Benes_mean",
    "Tot_Suplr_Benes_median",
    "Tot_Suplr_Benes_sum",
    "Tot_Suplr_Benes_std",
    "Tot_Suplr_Benes_min",
    "Tot_Suplr_Benes_max",
    "Tot_Suplr_Clms_mean",
    "Tot_Suplr_Clms_median",
    "Tot_Suplr_Clms_sum",
    "Tot_Suplr_Clms_std",
    "Tot_Suplr_Clms_min",
    "Tot_Suplr_Clms_max",
    "Tot_Suplr_Srvcs_mean",
    "Tot_Suplr_Srvcs_median",
    "Tot_Suplr_Srvcs_sum",
    "Tot_Suplr_Srvcs_std",
    "Tot_Suplr_Srvcs_min",
    "Tot_Suplr_Srvcs_max",
    "Avg_Suplr_Sbmtd_Chrg_mean",
    "Avg_Suplr_Sbmtd_Chrg_median",
    "Avg_Suplr_Sbmtd_Chrg_sum",
    "Avg_Suplr_Sbmtd_Chrg_std",
    "Avg_Suplr_Sbmtd_Chrg_min",
    "Avg_Suplr_Sbmtd_Chrg_max",
    "Avg_Suplr_Mdcr_Pymt_Amt_mean",
    "Avg_Suplr_Mdcr_Pymt_Amt_median",
    "Avg_Suplr_Mdcr_Pymt_Amt_sum",
    "Avg_Suplr_Mdcr_Pymt_Amt_std",
    "Avg_Suplr_Mdcr_Pymt_Amt_min",
    "Avg_Suplr_Mdcr_Pymt_Amt_max",
]

partb_aggregated_onehot_cols = [
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_Gndr",
]

partd_aggregated_onehot_cols = ["Prscrbr_Type"]

dmepos_aggregated_onehot_cols = [
    "Rfrg_Prvdr_Type",
    "Rfrg_Prvdr_Gndr",
]

# TODO - Non aggregated and new feature columns still need to be defined

file2columns = {
    "medicare-partb-2013-2019.csv.gz": None,
    "medicare-partb-aggregated-2013-2019.csv.gz": partb_aggregated_cols,
    "medicare-partb-aggregated-new-features-2013-2019.csv.gz": partb_aggregated_cols,
    "partb-aggregated-new-features-cleaned.csv.gz": partb_aggregated_cols,
    "partb-aggregated-new-features-noise.csv.gz": partb_aggregated_cols,
    "medicare-partd-2013-2019.csv.gz": None,
    "medicare-partd-aggregated-2013-2019.csv.gz": partd_aggregated_cols,
    "medicare-partd-aggregated-new-features-2013-2019.csv.gz": partd_aggregated_cols,
    "partd-aggregated-new-features-cleaned.csv.gz": partd_aggregated_cols,
    "partd-aggregated-new-features-noise.csv.gz": partd_aggregated_cols,
    "medicare-dmepos-2013-2019.csv.gz": None,
    "medicare-dmepos-aggregated-2013-2019.csv.gz": dmepos_aggregated_cols,
    "medicare-dmepos-aggregated-new-features-2013-2019.csv.gz": dmepos_aggregated_cols,
    "dmepos-aggregated-new-features-cleaned.csv.gz": dmepos_aggregated_cols,
    "dmepos-aggregated-new-features-noise.csv.gz": dmepos_aggregated_cols,
}

file2onehots = {
    "medicare-partb-2013-2019.csv.gz": None,
    "medicare-partb-aggregated-2013-2019.csv.gz": partb_aggregated_onehot_cols,
    "medicare-partb-aggregated-new-features-2013-2019.csv.gz": partb_aggregated_onehot_cols,
    "partb-aggregated-new-features-cleaned.csv.gz": partb_aggregated_onehot_cols,
    "partb-aggregated-new-features-noise.csv.gz": partb_aggregated_onehot_cols,
    "medicare-partd-2013-2019.csv.gz": None,
    "medicare-partd-aggregated-2013-2019.csv.gz": partd_aggregated_onehot_cols,
    "medicare-partd-aggregated-new-features-2013-2019.csv.gz": partd_aggregated_onehot_cols,
    "partd-aggregated-new-features-cleaned.csv.gz": partd_aggregated_onehot_cols,
    "partd-aggregated-new-features-noise.csv.gz": partd_aggregated_onehot_cols,
    "medicare-dmepos-2013-2019.csv.gz": None,
    "medicare-dmepos-aggregated-2013-2019.csv.gz": dmepos_aggregated_onehot_cols,
    "medicare-dmepos-aggregated-new-features-2013-2019.csv.gz": dmepos_aggregated_onehot_cols,
    "dmepos-aggregated-new-features-cleaned.csv.gz": dmepos_aggregated_onehot_cols,
    "dmepos-aggregated-new-features-noise.csv.gz": dmepos_aggregated_onehot_cols,
}


def get_dataset_key(input_file):
    if "partb" in input_file:
        return "partb"
    elif "partd" in input_file:
        return "partd"
    elif "dmepos" in input_file:
        return "dmepos"
    return "unknownfile"


def get_columns(input_file):
    filename = input_file.split("/")[-1]
    return file2columns.get(filename)


def get_onehot_columns(input_file):
    filename = input_file.split("/")[-1]
    return file2onehots.get(filename)


def load_cms_data(input_file, **kwargs):
    usecols = get_columns(input_file)
    logger.info(f"Loading data with columns\n{usecols}")

    df = pd.read_csv(input_file, usecols=usecols, **kwargs)
    logger.info(f"Loaded data with shape {df.shape}")

    onehot_cols = get_onehot_columns(input_file)
    logger.info(f"One-hot encoding columns:\n{onehot_cols}")
    df = pd.get_dummies(df, columns=onehot_cols)
    logger.info(f"One-hot encoded data, updated shape: {df.shape}")

    y, x = df["exclusion"], df.drop(columns=["exclusion"])

    return x, y


def add_noise_van_hulse(y, Lambda, Psi, neg_label=0, pos_label=1):
    """Adds class noise to y based on Knowledge discovery from imbalanced and noisy data by Jason Van Hulse et al and T.M. Khoshgoftaar.
    Args
        y (np.array): ground truth labels.
        Lambda (float): Class noise level percentage.
        Psi(float): Percentage of noise corresponding to the positive class.
        pos_label (any): positive class label.
        neg_label (any): negative class label.
    Returns
        noisy_y (np.array): transformed y with flipped class labels
    """
    pos_mask = y == pos_label
    pos_count = pos_mask.sum()

    noise_count = 2 * pos_count * Lambda
    pos_noise_count = math.floor(noise_count * Psi)
    neg_noise_count = int(noise_count - pos_noise_count)

    pos_indices = (y == pos_label).nonzero()[0]
    neg_indices = (y == neg_label).nonzero()[0]

    pos_noise_indices = np.random.choice(pos_indices, pos_noise_count, replace=False)
    neg_noise_indices = np.random.choice(neg_indices, neg_noise_count, replace=False)

    y[pos_noise_indices] = neg_label
    y[neg_noise_indices] = pos_label

    return y, pos_noise_count, neg_noise_count


def log_class_imbalance_levels(y):
    pos = (y == 1).sum() / len(y) * 100
    neg = (y == 0).sum() / len(y) * 100
    logger.info(f"Class imbalance levels: pos={pos}%, neg={neg}%")


def normalize_train_test(x_train, x_test):
    # normalize features
    scaler = MinMaxScaler()
    columns = x_train.columns
    x_train = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=columns)
    x_test = scaler.transform(x_test)
    x_test = pd.DataFrame(x_test, columns=columns)
    return x_train, x_test
