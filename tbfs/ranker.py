import logging

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

np.seterr(divide="ignore", invalid="ignore")

logger = logging.getLogger(__name__)

# Confusion matrix constants
CONFUSION_MATRIX = ["tn", "fp", "fn", "tp"]
TN_IDX, FP_IDX, FN_IDX, TP_IDX = 0, 1, 2, 3
INV_TN_IDX = FP_IDX
INV_FP_IDX = TN_IDX
INV_FN_IDX = TP_IDX
INV_TP_IDX = FN_IDX

# Ranking order
RANK_ASCENDING = ["gini-index", "deviance"]

metrics = [
    "f-score",
    "odds-ratio",
    "statistical-power",
    "probability-ratio",
    "gini-index",
    "kolmogorov-smirnov",
    "geometric-mean",
    "auc",
    "auprc",
    "mutual-info",
    "deviance",
    "matthews-correlation-coefficient",
]


class TBFSRanker:
    def __init__(self, t_delta=0.01, thresholds=None):
        """Create an instance of TBFS Feature Ranker.

        Parameters
        ----------
        t_delta: float
            Delta threshold used to enumerate thresholds for computing performance metrics.
        thresholds: list
            List of thresholds to use when performing fit. Overrides t_delta if not None.

        """
        self.t_delta = t_delta
        self.thresholds = np.arange(start=0, stop=1 + self.t_delta, step=self.t_delta)
        if thresholds is not None:
          self.thresholds = thresholds
        self.threshold_count = len(self.thresholds)
        logger.info(f"TBFSRanker initialized with t_delta {t_delta}")

    def fit(self, x, y):
        """Fit TBFS ranker to data set (x, y) and generate feature rankings.

        Parameters
        ----------
        x: pd.DataFrame
            Data set features
        y: pd.Series or np.array
            Data set labels

        Returns
        ----------
        rankings: pd.DataFrame
            Data frame of ranked features
        """

        # save copy of columns
        self.columns = x.columns

        # Normalize Data to [0, 1+t_delta]
        scaler = MinMaxScaler(feature_range=(0, 1 + self.t_delta))
        x = scaler.fit_transform(x)
        x = pd.DataFrame(x, columns=self.columns)
        logger.info("Normalized input features.")

        # record score for each (threshold, metric, feature)
        scores = np.zeros(
            shape=(self.threshold_count, len(self.columns), len(CONFUSION_MATRIX))
        )
        logger.info(f"Enumerating {self.threshold_count} thresholds.")
        for t_idx, t in enumerate(self.thresholds):
            # log percentage completed
            perc_complete = int(t_idx / self.threshold_count * 100)
            if perc_complete % 10 == 0 and perc_complete != 0:
                logger.info(f"{perc_complete} % of thresholds complete.")

            # compute predictions against the threhsold
            predictions = pd.DataFrame(np.where(x >= t, 1, 0), columns=self.columns)

            # record score for each (feature, metric) pair
            threshold_scores = np.zeros(
                shape=(len(self.columns), len(CONFUSION_MATRIX))
            )

            # iterate over columns
            for i, col in enumerate(self.columns):
                # record confusion matrix for each column
                tn, fp, fn, tp = confusion_matrix(y, predictions[col]).ravel()
                threshold_scores[i] = [tn, fp, fn, tp]
                # update scores with this threshold's scores
                scores[t_idx] = threshold_scores

        logger.info("Scores computed for all thresholds.")
        self.rankings = self._compute_rankings(scores)
        logger.info("Features rankings computed.")

    def to_csv(self, output_file):
        """Save rankings computed by fit method to csv file."""
        self.rankings.to_csv(output_file, index_label="feature")

    def from_csv(self, input_file):
        """Load rankings that were already computed"""
        self.rankings = pd.read_csv(input_file, index_col="feature")

    def top_k_features_by_metric(self, k, metric):
        """Return top K features for a metric."""
        ascending = metric in RANK_ASCENDING
        return (
            self.rankings.sort_values(by=metric, ascending=ascending)
            .head(k)
            .index.tolist()
        )

    def _compute_rankings(self, scores):
        """Helper method for computing feature ranking scores."""
        # true positive rates
        tprs = np.where(
            (scores[:, :, TP_IDX] + scores[:, :, FN_IDX]) == 0,
            0,
            scores[:, :, TP_IDX] / (scores[:, :, TP_IDX] + scores[:, :, FN_IDX]),
        )
        inv_tprs = np.where(
            (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FN_IDX]) == 0,
            0,
            scores[:, :, INV_TP_IDX]
            / (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FN_IDX]),
        )

        # true negative rates
        tnrs = np.where(
            (scores[:, :, TN_IDX] + scores[:, :, FP_IDX]) == 0,
            0,
            scores[:, :, TN_IDX] / (scores[:, :, TN_IDX] + scores[:, :, FP_IDX]),
        )
        inv_tnrs = np.where(
            scores[:, :, INV_TN_IDX] + scores[:, :, INV_FP_IDX] == 0,
            0,
            scores[:, :, INV_TN_IDX]
            / (scores[:, :, INV_TN_IDX] + scores[:, :, INV_FP_IDX]),
        )

        # false positive rates
        fprs = np.where(
            scores[:, :, FP_IDX] + scores[:, :, TN_IDX] == 0,
            0,
            scores[:, :, FP_IDX] / (scores[:, :, FP_IDX] + scores[:, :, TN_IDX]),
        )
        inv_fprs = np.where(
            scores[:, :, INV_FP_IDX] + scores[:, :, INV_TN_IDX] == 0,
            0,
            scores[:, :, INV_FP_IDX]
            / (scores[:, :, INV_FP_IDX] + scores[:, :, INV_TN_IDX]),
        )

        # precisions
        precisions = np.where(
            scores[:, :, TP_IDX] + scores[:, :, FP_IDX] == 0,
            0,
            scores[:, :, TP_IDX] / (scores[:, :, TP_IDX] + scores[:, :, FP_IDX]),
        )
        inv_precisions = np.where(
            scores[:, :, INV_TP_IDX] + scores[:, :, INV_FP_IDX] == 0,
            0,
            scores[:, :, INV_TP_IDX]
            / (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FP_IDX]),
        )

        # negative predictive values
        npvs = np.where(
            scores[:, :, TN_IDX] + scores[:, :, FN_IDX] == 0,
            0,
            scores[:, :, TN_IDX] / (scores[:, :, TN_IDX] + scores[:, :, FN_IDX]),
        )
        inv_npvs = np.where(
            scores[:, :, INV_TN_IDX] + scores[:, :, INV_FN_IDX] == 0,
            0,
            scores[:, :, INV_TN_IDX]
            / (scores[:, :, INV_TN_IDX] + scores[:, :, INV_FN_IDX]),
        )

        # Computer Ranking Metrics

        # ranker 1 - fsocre
        fscores = np.where(
            (precisions + tprs) == 0, 0, (2 * precisions * tprs) / (precisions + tprs)
        )

        inv_fscores = np.where(
            (inv_precisions + inv_tprs) == 0,
            0,
            (2 * inv_precisions * inv_tprs) / (inv_precisions + inv_tprs),
        )

        max_fscores = np.max(np.maximum(fscores, inv_fscores), axis=0)
        fscore_ranking = pd.Series(max_fscores, index=self.columns)

        # ranker 2 - odds ratio
        odds_ratios = np.where(
            scores[:, :, FP_IDX] * scores[:, :, FN_IDX] == 0,
            0,
            (scores[:, :, TP_IDX] * scores[:, :, TN_IDX])
            / (scores[:, :, FP_IDX] * scores[:, :, FN_IDX]),
        )

        inv_odds_ratios = np.where(
            (scores[:, :, INV_FP_IDX] * scores[:, :, INV_FN_IDX]) == 0,
            0,
            (scores[:, :, INV_TP_IDX] * scores[:, :, INV_TN_IDX])
            / (scores[:, :, INV_FP_IDX] * scores[:, :, INV_FN_IDX]),
        )

        max_odds_ratios = np.max(np.maximum(odds_ratios, inv_odds_ratios), axis=0)
        odds_ratio_ranking = pd.Series(max_odds_ratios, index=self.columns)

        # ranker 3 - power
        powers = (1 - fprs) ** 5 - (1 - tprs) ** 5
        inv_powers = (1 - inv_fprs) ** 5 - (1 - inv_tprs) ** 5

        max_powers = np.max(np.maximum(powers, inv_powers), axis=0)
        power_ranking = pd.Series(max_powers, index=self.columns)

        # ranker 4 - probability ratios
        prob_ratios = np.where(fprs == 0, 0, tprs / fprs)
        inv_prob_ratios = np.where(inv_fprs == 0, 0, inv_tprs / inv_fprs)

        max_prob_ratios = np.max(np.maximum(prob_ratios, inv_prob_ratios), axis=0)
        prob_ratio_ranking = pd.Series(max_prob_ratios, index=self.columns)

        # ranker 5 - gini index
        gini_indices = 2 * precisions * (1 - precisions) + 2 * npvs * (1 - npvs)
        inv_gini_indices = 2 * inv_precisions * (1 - inv_precisions) + 2 * inv_npvs * (
            1 - inv_npvs
        )

        min_gini_indices = np.min(np.minimum(gini_indices, inv_gini_indices), axis=0)
        gini_index_ranking = pd.Series(min_gini_indices, index=self.columns)

        # ranker 6 - Kolmogorov-Smirnov
        ks = np.abs(tprs - fprs)
        inv_ks = np.abs(inv_tprs - inv_fprs)

        max_ks = np.max(np.maximum(ks, inv_ks), axis=0)
        ks_ranking = pd.Series(max_ks, index=self.columns)

        # ranker 7 - Geometric Mean
        gmeans = (tprs * tnrs) ** 0.5
        inv_gmeans = (inv_tprs * inv_tnrs) ** 0.5

        max_gmeans = np.max(np.maximum(gmeans, inv_gmeans), axis=0)
        gmean_ranking = pd.Series(max_gmeans, index=self.columns)

        # ranker 8 - AUC
        aucs = np.zeros(len(self.columns))
        inv_aucs = np.zeros(len(self.columns))

        for i in range(len(self.columns)):
            aucs[i] = auc(tprs[:, i], fprs[:, i])
            inv_aucs[i] = auc(inv_tprs[:, i], inv_fprs[:, i])

        max_aucs = np.maximum(aucs, inv_aucs)
        auc_ranking = pd.Series(max_aucs, index=self.columns)

        # ranker 9 - auprcs
        auprcs = np.zeros(len(self.columns))
        inv_auprcs = np.zeros(len(self.columns))

        for i in range(len(self.columns)):
            auprcs[i] = auc(tprs[:, i], precisions[:, i])
            inv_auprcs[i] = auc(inv_tprs[:, i], inv_precisions[:, i])

        max_auprcs = np.maximum(auprcs, inv_auprcs)
        auprc_ranking = pd.Series(max_auprcs, index=self.columns)

        # ranker 10 - mi
        mi_1 = np.where(
            scores[:, :, FP_IDX] == 0,
            0,
            scores[:, :, FP_IDX]
            * np.log10(
                scores[:, :, FP_IDX]
                / (
                    (scores[:, :, TP_IDX] + scores[:, :, FP_IDX])
                    * (scores[:, :, FP_IDX] + scores[:, :, TN_IDX])
                )
            ),
        )

        mi_2 = np.where(
            scores[:, :, TP_IDX] == 0,
            0,
            scores[:, :, TP_IDX]
            * np.log10(
                scores[:, :, TP_IDX]
                / (
                    (scores[:, :, TP_IDX] + scores[:, :, FP_IDX])
                    * (scores[:, :, TP_IDX] + scores[:, :, FN_IDX])
                )
            ),
        )

        mi_3 = np.where(
            scores[:, :, TN_IDX] == 0,
            0,
            scores[:, :, TN_IDX]
            * np.log10(
                scores[:, :, FN_IDX]
                / (
                    (scores[:, :, FN_IDX] + scores[:, :, TN_IDX])
                    * (scores[:, :, FP_IDX] + scores[:, :, TN_IDX])
                )
            ),
        )

        mi_4 = np.where(
            scores[:, :, FN_IDX] == 0,
            0,
            scores[:, :, FN_IDX]
            * np.log10(
                scores[:, :, FN_IDX]
                / (
                    (scores[:, :, FN_IDX] + scores[:, :, TN_IDX])
                    * (scores[:, :, TP_IDX] + scores[:, :, FN_IDX])
                )
            ),
        )

        mi_indices = mi_1 + mi_2 + mi_3 + mi_4

        inv_mi_1 = np.where(
            scores[:, :, INV_FP_IDX] == 0,
            0,
            scores[:, :, INV_FP_IDX]
            * np.log10(
                scores[:, :, INV_FP_IDX]
                / (
                    (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FP_IDX])
                    * (scores[:, :, INV_FP_IDX] + scores[:, :, INV_TN_IDX])
                )
            ),
        )

        inv_mi_2 = np.where(
            scores[:, :, INV_TP_IDX] == 0,
            0,
            scores[:, :, INV_TP_IDX]
            * np.log10(
                scores[:, :, INV_TP_IDX]
                / (
                    (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FP_IDX])
                    * (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FN_IDX])
                )
            ),
        )

        inv_mi_3 = np.where(
            scores[:, :, INV_TN_IDX] == 0,
            0,
            scores[:, :, INV_TN_IDX]
            * np.log10(
                scores[:, :, INV_FN_IDX]
                / (
                    (scores[:, :, INV_FN_IDX] + scores[:, :, INV_TN_IDX])
                    * (scores[:, :, INV_FP_IDX] + scores[:, :, INV_TN_IDX])
                )
            ),
        )

        inv_mi_4 = np.where(
            scores[:, :, INV_FN_IDX] == 0,
            0,
            scores[:, :, INV_FN_IDX]
            * np.log10(
                scores[:, :, INV_FN_IDX]
                / (
                    (scores[:, :, INV_FN_IDX] + scores[:, :, INV_TN_IDX])
                    * (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FN_IDX])
                )
            ),
        )

        inv_mi_indices = inv_mi_1 + inv_mi_2 + inv_mi_3 + inv_mi_4

        max_mi_indices = np.max(np.maximum(mi_indices, inv_mi_indices), axis=0)
        mi_index_ranking = pd.Series(max_mi_indices, index=self.columns)

        # ranker 11 - dev
        dev = (scores[:, :, TP_IDX] - precisions) ** 2 + (
            scores[:, :, FN_IDX] - (1 - npvs)
        ) ** 2
        inv_dev = (scores[:, :, INV_TP_IDX] - inv_precisions) ** 2 + (
            scores[:, :, INV_FN_IDX] - (1 - inv_npvs)
        ) ** 2

        min_dev = np.min(np.minimum(dev, inv_dev), axis=0)
        dev_ranking = pd.Series(min_dev, index=self.columns)

        # ranker 12 - mcc
        mcc = np.where(
            (scores[:, :, TP_IDX] + scores[:, :, FP_IDX])
            * (scores[:, :, TP_IDX] + scores[:, :, FN_IDX])
            * (scores[:, :, TN_IDX] + scores[:, :, FP_IDX])
            * (scores[:, :, TN_IDX] + scores[:, :, FN_IDX])
            == 0,
            0,
            (
                scores[:, :, TP_IDX] * scores[:, :, TN_IDX]
                - scores[:, :, FP_IDX] * scores[:, :, FN_IDX]
            )
            / np.power(
                (scores[:, :, TP_IDX] + scores[:, :, FP_IDX])
                * (scores[:, :, TP_IDX] + scores[:, :, FN_IDX])
                * (scores[:, :, TN_IDX] + scores[:, :, FP_IDX])
                * (scores[:, :, TN_IDX] + scores[:, :, FN_IDX]),
                0.5,
            ),
        )

        inv_mcc = np.where(
            (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FP_IDX])
            * (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FN_IDX])
            * (scores[:, :, INV_TN_IDX] + scores[:, :, INV_FP_IDX])
            * (scores[:, :, INV_TN_IDX] + scores[:, :, INV_FN_IDX])
            == 0,
            0,
            (
                scores[:, :, INV_TP_IDX] * scores[:, :, INV_TN_IDX]
                - scores[:, :, INV_FP_IDX] * scores[:, :, INV_FN_IDX]
            )
            / np.power(
                (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FP_IDX])
                * (scores[:, :, INV_TP_IDX] + scores[:, :, INV_FN_IDX])
                * (scores[:, :, INV_TN_IDX] + scores[:, :, INV_FP_IDX])
                * (scores[:, :, INV_TN_IDX] + scores[:, :, INV_FN_IDX]),
                0.5,
            ),
        )

        max_mcc = np.max(np.maximum(mcc, inv_mcc), axis=0)
        mcc_ranking = pd.Series(max_mcc, index=self.columns)

        return pd.DataFrame(
            {
                "f-score": fscore_ranking,
                "odds-ratio": odds_ratio_ranking,
                "statistical-power": power_ranking,
                "probability-ratio": prob_ratio_ranking,
                "gini-index": gini_index_ranking,
                "kolmogorov-smirnov": ks_ranking,
                "geometric-mean": gmean_ranking,
                "auc": auc_ranking,
                "auprc": auprc_ranking,
                "mutual-info": mi_index_ranking,
                "deviance": dev_ranking,
                "matthews-correlation-coefficient": mcc_ranking,
            }
        )
