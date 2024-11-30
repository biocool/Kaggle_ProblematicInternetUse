import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def check_range_values(x_df, y_df=None, feature_explanation_df=None, label_col=None, output_path=None, mode=None):

    num_samples = x_df.shape[0]
    num_features = x_df.shape[1]

    categorical_columns = [col for col in x_df.columns if x_df[col].dtype == 'object' or
                           x_df[col].dtype.name == 'category']
    numerical_columns = [col for col in x_df.columns if col not in categorical_columns]

    num_categorical_cols = len(categorical_columns)
    num_numerical_columns = len(numerical_columns)

    min_values_list = []
    max_values_list = []
    avg_values_list = []
    std_values_list = []
    variance_values_list = []
    median_value_list = []
    num_na_value_list = []
    instrument_list = []
    description_list = []
    type_list = []
    values_list = []
    value_labels_list = []

    for numerical_col in numerical_columns:

        further_info = feature_explanation_df.loc[feature_explanation_df['Field'] == numerical_col]

        min_val = x_df[numerical_col].min()
        max_val = x_df[numerical_col].max()
        avg_val = x_df[numerical_col].mean()
        std_val = x_df[numerical_col].std()
        var_val = x_df[numerical_col].var()
        median_val = x_df[numerical_col].median()
        num_na = x_df[numerical_col].isna().sum()

        min_values_list.append(min_val)
        max_values_list.append(max_val)
        avg_values_list.append(avg_val)
        std_values_list.append(std_val)
        variance_values_list.append(var_val)
        median_value_list.append(median_val)
        num_na_value_list.append(num_na)

        instrument_list.append(further_info['Instrument'].values[0])
        description_list.append(further_info['Description'].values[0])
        type_list.append(further_info['Type'].values[0])
        values_list.append(further_info['Values'].values[0])
        value_labels_list.append(further_info['Value Labels'].values[0])

    summary_df_numerical_cols = pd.DataFrame({'Numerical Feature name': numerical_columns, 'min': min_values_list,
                                              'max': max_values_list, 'average': avg_values_list,
                                              'std': std_values_list, 'variance': variance_values_list,
                                              'median': median_value_list, '# missing values': num_na_value_list,
                                              '# total number of values': x_df.shape[0],
                                              "Instrument": instrument_list, "Description": description_list,
                                              "Type": type_list, "Values": values_list,
                                              "Values label": value_labels_list
                                              })

    num_unique_values_list = []
    num_na_value_list = []
    imbalance_ratio_list = []
    instrument_list = []
    description_list = []
    type_list = []
    values_list = []
    value_labels_list = []

    for categorical_column in categorical_columns:

        further_info = feature_explanation_df.loc[feature_explanation_df['Field'] == categorical_column]

        num_unique_values = x_df[categorical_column].dropna().nunique()
        num_na = x_df[categorical_column].isna().sum()

        value_counts = x_df[categorical_column].dropna().value_counts()
        imbalance_ratio = value_counts.max() / value_counts.min()

        num_unique_values_list.append(num_unique_values)
        num_na_value_list.append(num_na)
        imbalance_ratio_list.append(imbalance_ratio)

        instrument_list.append(further_info['Instrument'].values[0])
        description_list.append(further_info['Description'].values[0])
        type_list.append(further_info['Type'].values[0])
        values_list.append(further_info['Values'].values[0])
        value_labels_list.append(further_info['Value Labels'].values[0])

    if y_df is not None:
        num_unique_values = y_df[label_col].dropna().nunique()
        num_na = y_df[label_col].isna().sum()

        value_counts = y_df.dropna().value_counts()
        imbalance_ratio = value_counts.max() / value_counts.min()

        num_unique_values_list.append(num_unique_values)
        num_na_value_list.append(num_na)
        imbalance_ratio_list.append(imbalance_ratio)

        categorical_columns.append(label_col)

        instrument_list.append(np.nan)
        description_list.append(np.nan)
        type_list.append(np.nan)
        values_list.append(np.nan)
        value_labels_list.append(np.nan)

        if output_path is not None:
            # Plot box plot for distribution of values
            # Prepare data for box plot
            value_distribution_df = y_df[label_col].value_counts().reset_index()
            value_distribution_df.columns = ['Category', 'Count']

            # Plot box plot for counts of each value
            # Plot bar chart for counts of each value
            plt.figure(figsize=(12, 6))
            sns.barplot(data=value_distribution_df, x='Category', y='Count')
            plt.title(f"Bar Chart for {label_col} Value Distribution")
            plt.xlabel("Category")
            plt.ylabel("Count")
            plt.tight_layout()  # Adjust layout for better display
            plt.savefig(output_path + '/' + mode + '.' + label_col + '.distribution.jpg', dpi=300)

    summary_df_categorical_cols = pd.DataFrame({'Categorical Feature name': categorical_columns,
                                                '# unique values': num_unique_values_list,
                                                'imbalance ratio': imbalance_ratio_list,
                                                '# missing values': num_na_value_list,
                                                '# total number of values': x_df.shape[0],
                                                "Instrument": instrument_list, "Description": description_list,
                                                "Type": type_list, "Values": values_list,
                                                "Values label": value_labels_list})

    if output_path is not None:
        summary_df_numerical_cols.to_csv(output_path + '/' + mode + '.numerical_cols.summery.csv', index=False)
        summary_df_categorical_cols.to_csv(output_path + '/' + mode + '.categorical_cols.summery.csv', index=False)

        # Plot
        # Data for the bar plot
        labels = ['# Samples', '# Features', '# Categorical Columns', '# Numerical Columns']
        values = [num_samples, num_features, num_categorical_cols, num_numerical_columns]
        fig, ax = plt.subplots(figsize=(10, 6))

        x_positions = np.arange(len(labels))  # X positions for the bars
        bar_width = 0.6  # Bar width to leave space between bars

        # Create bars
        bars = ax.bar(x_positions, values, width=bar_width, alpha=0.8, edgecolor='black')

        # Add value labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, str(yval), ha='center', va='bottom', fontsize=10)

        # Customize the chart
        ax.set_title(mode + ' Data Overview', fontsize=14)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=11)
        ax.grid(axis='y', alpha=0.4)

        # Adjust layout for better spacing
        plt.tight_layout()

        plt.savefig(output_path + '/' + mode + '.summary.space.jpg', dpi=300)


def diff_train_test_feature_space(train_dat, test_dat, label_col = 'sii', output_path=None):

    train_dat_cols = set([v for v in train_dat.columns.tolist() if v != label_col])
    test_dat_cols = set(test_dat.columns.tolist())

    cols_in_train_not_test = list(train_dat_cols - test_dat_cols)
    cols_in_test_not_train = list(test_dat_cols - train_dat_cols)

    # Create a DataFrame with equal length columns
    max_len = max(len(cols_in_train_not_test), len(cols_in_test_not_train))
    cols_in_train_not_test.extend([None] * (max_len - len(cols_in_train_not_test)))
    cols_in_test_not_train.extend([None] * (max_len - len(cols_in_test_not_train)))

    df_diff = pd.DataFrame({
        'Columns in Train data not in Test': cols_in_train_not_test,
        'Columns in Test data not in Train': cols_in_test_not_train
    })

    if output_path is not None:
        df_diff.to_csv(output_path + '/compare_train_test_feature_space.csv', index=False)

    return df_diff

