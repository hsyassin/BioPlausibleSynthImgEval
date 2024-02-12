import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


__author__ = "Hadya Yassin"
__maintainer__ = "Hadya Yassin"
__email__ = "hadya.yassin@hpi.de"
__status__ = "Production"
__copyright__ = "Copyright (c) 2023 Yassin, Hadya Sa'ad Abdalwadoud"
__license__ = "MIT"

"""
This code defines a Python class Stat_Test designed for conducting various statistical tests and analyses on datasets, 
particularly focusing on comparing real and synthetic values across specific regions. It is structured to work with datasets 
that have at least two types of values (real and synthetic) and aims to evaluate differences between these groups through statistical methods.
It also has additional function such as outlier detection and normality assumbtion check"""



class Stat_Test:
    def __init__(self, real_values, syn_values):
        self.real_values, self.syn_values = real_values, syn_values


        
    @staticmethod
    def outliers(data, region, cov, TYP):

        df = data[data['Image_Type'] == 'Real'] if TYP == "Real" else data[data['Image_Type'] == 'Synthetic']

        # Z-score method
        z_scores = stats.zscore(df[f'{region}_scaled_Area'])
        outliers_z = df[f'{region}_scaled_Area'][(z_scores < -3) | (z_scores > 3)]

        # IQR method
        Q1 = df[f'{region}_scaled_Area'].quantile(0.25)
        Q3 = df[f'{region}_scaled_Area'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = df[f'{region}_scaled_Area'][(df[f'{region}_scaled_Area'] < (Q1 - 1.5 * IQR)) | (df[f'{region}_scaled_Area'] > (Q3 + 1.5 * IQR))]

        tst_typs = ['z', 'iqr']

        

        for tst_typ in tst_typs:

            new_df = pd.DataFrame({})

            outliers_tst_typ = outliers_z if tst_typ == 'z' else outliers_iqr
            # print(outliers_tst_typ)

            for idx in outliers_tst_typ.index:
                reg_scaled_Aarea = outliers_tst_typ[idx]

                try:
                    out_row = data[data[f'{region}_scaled_Area'] == reg_scaled_Aarea]
                except:
                    raise RuntimeError('dsdsdsdsd')

                new_df = new_df.append(out_row, ignore_index=True)  # Append the matching row to 'df'
            
            if tst_typ == 'z':
                new_df['tst_typ'] = 'z'
                outliers_z = new_df
                
            elif tst_typ == 'iqr':
                new_df['tst_typ'] = 'iqr'
                outliers_iqr = new_df


        # print(outliers_z, outliers_iqr)

        return outliers_z, outliers_iqr

    def cohens_d(self):
        x, y = self.real_values, self.syn_values
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    @staticmethod
    def CI_cohens_d(effect_size, n1, n2):
        # Calculate Standard Error
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + (effect_size ** 2 / (2 * (n1 + n2))))

        # Determine Z-score for 95% CI
        z_score = norm.ppf(0.975)  # two-tailed for 95%

        # Calculate Confidence Interval
        ci_lower = effect_size - z_score * se_d
        ci_upper = effect_size + z_score * se_d
        
        return ci_lower, ci_upper
    
    def permutation_test(self, n_permutations=1000):
        observed_stat = np.mean(self.real_values) - np.mean(self.syn_values)
        combined = np.concatenate([self.real_values, self.syn_values])
        count = 0

        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_real = combined[:len(self.real_values)]
            perm_syn = combined[len(self.real_values):]
            perm_stat = np.mean(perm_real) - np.mean(perm_syn)

            if abs(perm_stat) >= abs(observed_stat):
                count += 1

        p_value = count / n_permutations
        return observed_stat, p_value

    def check_data(self, region, test_data, s_type, cov, ratio):
        # Shapiro-Wilk Test for normality
  
        shapiro_real = stats.shapiro(self.real_values)
        shapiro_syn = stats.shapiro(self.syn_values)

        # Histogram for distribution (optional)
        plt.figure(figsize=(10, 6))
        sns.histplot(self.real_values, kde=True, color='blue', label='Real')
        sns.histplot(self.syn_values, kde=True, color='orange', label='Synthetic')
        plt.title('Histogram of Real and Synthetic Values')
        plt.legend()
        plt.savefig(f'/Dist_Histo/{region}_{test_data}_{s_type}_{cov}_{ratio}ratio_histo_realandgen.png', dpi=100)
        plt.show()
        plt.close()


        # Skewness
        skewness_real = stats.skew(self.real_values)
        skewness_syn = stats.skew(self.syn_values)
        

        return shapiro_real, shapiro_syn, skewness_real, skewness_syn


    @staticmethod
    def flag_p_value(p_value):
        """Define a function to flag p-values with stars based on their significance levels"""
        if p_value < 0.001:
            return f"{p_value:.1e} < 1e-3"               
        elif p_value < 0.01:
            return f"{p_value:.1e} < 1e-2"                
        elif p_value < 0.05:
            return f"{p_value:.1e} < 5e-2"               
        else:
            return f"{p_value:.1e}"                


def main(data, real_values, syn_values, analysis_type, region, test_data, s_type, cov, ratio):

    s = Stat_Test(real_values, syn_values)

    #######1- A test for normality; it checks whether a sample comes from a normally distributed population. It's widely used for checking the assumption of normality in data.
    shapiro_real, shapiro_syn, skewness_real, skewness_syn = s.check_data(region, test_data, s_type, cov, ratio)

    
    # Print the results, This is the test statistic value. It's a measure of how much the data deviates from a normal distribution. In a perfect normal distribution, this value would be close to 1. The closer the statistic is to 1, the more likely the data follows a normal distribution.
    print("Shapiro-Wilk Test for Real Values") 
    print(f"Statistic: {shapiro_real.statistic:.1e}, P-value: {shapiro_syn.pvalue:.1e}")

    print("\nShapiro-Wilk Test for Synthetic Values")
    print(f"Statistic: {shapiro_syn.statistic:.1e}, P-value: {shapiro_syn.pvalue:.1e}")

    # Measures the asymmetry of the probability distribution of a real-valued random variable. Positive skew indicates a distribution with an asymmetric tail extending towards more positive values.
    print(f"Skewness for Real Values: {skewness_real:.1e}")
    print(f"Skewness for Synthetic Values: {skewness_syn:.1e}")

    #####################

    #########2- Effect Size Cohen An effect size used to indicate the standardized difference between two means. It's a measure of how separated or overlapping two distributions are.
    effect_size = s.cohens_d()
    print(f'Effect Size (Cohen\'s d):{effect_size:.1e}')

    # This gives a range of values for Cohen's d that are likely to contain the true effect size with a certain level of confidence (usually 95%). It provides an estimate of the uncertainty or variability of the effect size.
    cohen_ci_lower, cohen_ci_upper = s.CI_cohens_d(effect_size, len(real_values), len(syn_values))
    print("95% Confidence Interval for Cohen's d:", cohen_ci_lower, cohen_ci_upper)

    #############################################

    ############### Permutation test, also known as a randomization test, a non-parametric method to determine if two samples are from the same distribution. Useful when assumptions for parametric tests are not met. It involves shuffling data and recalculating a test statistic multiple times to estimate a p-value.
    observed_stat, perm_p_value = s.permutation_test()
    print(f'Permutation Test Statistic: {observed_stat:.1e}, P-value: {perm_p_value:.1e}')
    ###################

    #########4- MWU A non-parametric test used to determine if there are differences between two independent samples. It's an alternative to the t-test for non-normally distributed data.
    mwu_stat, p_value = stats.mannwhitneyu(real_values, syn_values)
    print(f'MWU statt: {mwu_stat:.1e}. MWU p_value:{p_value:.1e}')
    ############################3

    ################KS#############
    # Perform KS Test A non-parametric test that determines if two samples are drawn from the same distribution. It compares the cumulative distributions of two datasets.
    ks_stat, ks_p_value = stats.ks_2samp(real_values, syn_values)

    print(f'KS Statistic: {ks_stat:.1e}. KS P-value: {ks_p_value:.1e}')
    ####################################

    #####outliers identifier    
    out_z_Real, out_iqr_Real = s.outliers(data, region, cov, TYP = "Real", area_typ = 'scaled')
    out_z_Syn, out_iqr_Syn = s.outliers(data, region, cov, TYP = "Syn", area_typ = 'scaled')
    ######

    
    ks_pvalue = s.flag_p_value(ks_p_value)
    permut_pvalue = s.flag_p_value(perm_p_value)
    cohens_d = effect_size
    ci_lower = cohen_ci_lower
    ci_upper = cohen_ci_upper


    # Prepare data for saving to CSV
    data_to_save = {
        "analysis_type": analysis_type,
        "region": region,
        "ratio":ratio,
        "sampling type": s_type,
        "sample cov": cov, 
        "Permut_test stat": observed_stat, 
        "Permut pvalue": permut_pvalue,
        "MWU Stat": mwu_stat,
        "MWU P-value": s.flag_p_value(p_value),
        "KS Statistic": ks_stat,
        "KS P-value": ks_pvalue,
        "Cohen's d": f'{cohens_d:.1e}',
        "cohen ci lower": f"{ci_lower:.1e}",
        "cohen ci upper": f"{ci_upper:.1e}",
        "Shapiro-Wilk pvalue Real": s.flag_p_value(shapiro_real.pvalue),
        "skewness Real": skewness_real,
        "Shapiro-Wilk pvalue Syn": s.flag_p_value(shapiro_syn.pvalue),
        "skewness Syn": skewness_syn,
        "Shapiro-Wilk stat Real": shapiro_real.statistic,
        "Shapiro-Wilk stat Syn": shapiro_syn.statistic,
        "data": test_data
    }

    # Convert to DataFrame
    df_test = pd.DataFrame([data_to_save])

    # Define file path
    folder = "5Synthetic-1Real" 
    file_path = f'final_code/{folder}/Stats/5-1Samples-scalednoZero_Statistical_test_results.csv'

    # Check if the file exists and set the header accordingly
    file_exists = os.path.exists(file_path)

    # # Save to CSV
    # # Append to CSV if it exists, otherwise create a new one
    df_test.to_csv(file_path, mode='a', index=False, header=not file_exists)            

    df_test.head()  # Display the first few rows of the DataFrame for confirmation


    return ks_pvalue, permut_pvalue, cohens_d, ci_lower, ci_upper, out_z_Real, out_iqr_Real, out_z_Syn, out_iqr_Syn




if __name__ == '__main__':
    # Sample data for recalculating statistics
    np.random.seed(0)
    real_values = np.random.normal(0, 1, 100)
    syn_values = np.random.normal(0.5, 1, 100)
    main()



print("here")