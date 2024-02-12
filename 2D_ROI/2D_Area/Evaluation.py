import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Stat_tests import main



__author__ = "Hadya Yassin"
__maintainer__ = "Hadya Yassin"
__email__ = "hadya.yassin@hpi.de"
__status__ = "Production"
__copyright__ = "Copyright (c) 2023 Yassin, Hadya Sa'ad Abdalwadoud"
__license__ = "MIT"


"""

This code performs a detailed comparative analysis between real and synthetic MRI imaging data, with an emphasis on distinct brain regions 
while considering a range of covariates. It achieves this through two primary approaches:

1-  Visual Analysis: It generates box plots for univariate data visualization, offering a clear visual comparison between a subset of the real data
    and its corresponding synthetic counterpart, whether the distribution is matching or not. This visual representation helps in identifying differences or similarities in the distribution of measured areas across the specified brain regions.

2-  Quantitative Analysis: The code constructs dataframes dedicated to encapsulating the results of statistical analyses conducted between the two groups under comparison.
    These dataframes serve as a structured repository for quantitative insights, facilitating a deeper understanding of the statistical significance and effect sizes that distinguish real data from synthetic ones.
    
"""


regions = ['LV', 'H']

replicate = False
discard = True
ratio = '5Syn-1Real'
network = "D_Unet"          ## netword used in segmentation of ROIs
exclude_outliers = True     ###exclude areas at extreme ages < 60 and ages > 90  

test_data = "Org_adni"                ####"adni", "Org_Adni"
covariates =  ['', 'Age','CDGLOBAL','Sex'] 
plot = 'box'


for cov in covariates:
    
    if cov == '':
        doms = ['']
    elif 'Age' in cov:
        doms = ['Low', 'High']
    elif 'Sex' in cov:
        doms = ['F5', 'M5', 'F', 'M', "F15", 'M15']
    elif 'CDGLOBAL' in cov:
        doms = ['CN', 'AD']

    

    for dom in doms:
        analysis_type = f"Univariate-{dom}"
        s_type = f"matched" if cov == "" else f"unmatched-{cov}-{dom}"

        df = pd.read_csv(f'/data/Areas_5Samples/{ratio}Samples/...{s_type}{cov}{dom}.csv') 
        df.columns = df.columns.str.strip()

        for region in regions:
            ## get the magnitude of areas and discard 0 values
            df[f'{region}_scaled_Area'] = df[f'{region}_scaled_Area'].abs()
            df = df[df[f'{region}_scaled_Area'] != 0]

            non_positive_count = len(df[df[f'{region}_scaled_Area'] <= 0])

            if non_positive_count == 0:
                print("All values are positive")
            else:
                print("There are non-positive values")

        # Filter data to exclude outliers
        if exclude_outliers:
            df = df[(df['Age'] >= 60) & (df['Age'] <= 90)]
            ex = "_only60-90"

        # Set cdr values above 1 to 1
        df['CDGLOBAL'] = df['CDGLOBAL'].apply(lambda x: 1 if x >= 1 else x)


        # Set overall aesthetics
        sns.set_style("whitegrid")
        # Create a subplot of 1 row and 2 columns (one for each region)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        fig.suptitle('Real vs. Synthetic Area Comparison', fontsize=24)


        # Iterate through the areas and their corresponding axis
        for ax, area in zip(axes, ['LV', 'H']):
        # for ax, (area, values) in zip(axes, data.items()):

            palette_dict = {
                'Real': 'cornflowerblue',
                'Synthetic': 'coral'
            }

            

            sns.set_palette(palette_dict.values())

            if plot == 'Violin':
                sns.violinplot(x='Image_Type', y=f'{area}_scaled_Area', data=df, palette=palette_dict, ax=ax)
                # Calculate and replot medians
                for i, img_type in enumerate(df['Image_Type'].unique()):
                    median = df[df['Image_Type'] == img_type][f'{area}_scaled_Area'].median()
                    ax.plot(i, median, 'o', color='white', markersize=10)
            elif plot == 'Box':
                # Use 'hue' to create side by side Box plots for 'Real' and 'Synthetic'
                sns.boxplot(x='Image_Type', y=f'{area}_scaled_Area', data=df, ax=ax, palette=palette_dict)      ##, hue='Image_Type'

                
            name2 = "b) Hippocampus"if area == "H" else "a) Ventricles"

            ax.set_title(name2, fontsize=22)
            ax.set_ylabel(f'Area', fontsize=20)
            ax.set_xlabel('')
            ax.tick_params(axis='both', which='major', labelsize=20)

            if discard:

                real_values = df[df['Image_Type'] == 'Real'][f'{area}_scaled_Area']
                syn_values = df[df['Image_Type'] == 'Synthetic'][f'{area}_scaled_Area']

                ###All relevant statistical tests and parameters 
                ks_pvalue, permut_pvalue, cohens_d, ci_lower, ci_upper = main(df, real_values, syn_values, analysis_type, area, test_data, s_type, cov, ratio)


            ###(Optional) add statuistical findings to figures

            # color1 = 'r' if "<" in ks_pvalue else 'g'
            # color2 = 'r' if "<" in permut_pvalue else 'g'
            # color3 = 'r' if ci_lower < 0 and ci_upper < 0 or ci_lower > 0 and ci_upper > 0 else 'g'

            # # Create custom legend entries
            # legend_elements = [
            #     Line2D([0], [0], marker='o', color='w', label=f'KS P-value: {ks_pvalue}',
            #         markerfacecolor=color1, markersize=15),
            #     Line2D([0], [0], marker='o', color='w', label=f'Permutation P-value: {permut_pvalue}',
            #         markerfacecolor=color2, markersize=15),
            #     Line2D([0], [0], marker='o', color='w', label=f'Cohen\'s d: {cohens_d:.3e} (95% CI: [{ci_lower:.3e}, {ci_upper:.3e}])',
            #         markerfacecolor=color3, markersize=15),
            # ]

            # # Add the legend under the graph with a specific location
            # ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.10), fontsize=16)



        plt.tight_layout()
        # plt.grid(False)
        plt.savefig(f'/results/Univariate_comp/{s_type}_{analysis_type}-{ratio}ratio-{plot}Plots_RealVsSyn.png', dpi=100)
        plt.show()
        plt.close()
