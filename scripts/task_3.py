
from scipy.stats import ttest_ind

class ABHypothesisTesting:
    """
    Class to perform A/B Hypothesis Testing.

    Methods:
        __init__: Initializes the class with the dataset.
        create_ab_groups: Segments data into control and test groups based on a feature.
        perform_t_test: Conducts a t-test on a specific metric between two groups.
    """
    def __init__(self, data):
        self.data = data

    def create_ab_groups(self, feature, value1, value2):
        """
        Splits the data into two groups based on feature values.
        Args:
            feature (str): The feature to split on.
            value1: Value for group A.
            value2: Value for group B.
        Returns:
            Tuple: DataFrames for group A and group B.
        """
        group_a = self.data[self.data[feature] == value1]
        group_b = self.data[self.data[feature] == value2]
        return group_a, group_b

    def perform_t_test(self, group_a, group_b, metric):
        """
        Conducts a t-test on a specific metric between two groups.
        Args:
            group_a (DataFrame): Control group.
            group_b (DataFrame): Test group.
            metric (str): Metric to test.
        Returns:
            Tuple: t-statistic and p-value.
        """
        stat, p_value = ttest_ind(group_a[metric], group_b[metric], nan_policy='omit')
        return stat, p_value


