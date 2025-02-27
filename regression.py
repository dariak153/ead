import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def exclude_outliers(df, column, upper_limit, lower_limit=None):
    if column not in df.columns:
        print(f"Kolumna '{column}' nie istnieje w zbiorze danych.\n")
        return df

    if lower_limit is not None:
        condition = (df[column] >= lower_limit) & (df[column] <= upper_limit)
    else:
        condition = df[column] <= upper_limit
    filtered_df = df[condition]
    removed = df.shape[0] - filtered_df.shape[0]
    #print(f"Usunięto {removed} wartości odstających z kolumny '{column}'")
    #print(f"Rozmiar danych: {filtered_df.shape}\n")
    return filtered_df

def plot(df, model, predictor, response):
    residuals = model.predict(df) - df[response]
    df_plot = df.copy()
    df_plot['Prediction'] = model.predict(df_plot)
    df_plot = df_plot.sort_values(by=predictor)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.scatterplot(x=predictor, y=response, data=df_plot, ax=axes[0, 0], label='Rzeczywiste')
    sns.lineplot(x=predictor, y='Prediction', data=df_plot, color='red', ax=axes[0, 0], label='Przewidywane')
    axes[0, 0].set_title(f'{predictor} vs {response}')

    sns.scatterplot(x=df_plot[predictor], y=residuals, ax=axes[0, 1])
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel(predictor)
    axes[0, 1].set_ylabel('Residual values')

    sns.histplot(residuals, kde=True, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Residual values')
    axes[1, 0].set_ylabel('Frequency')

    sm.qqplot(residuals, line='s', ax=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')

    plt.tight_layout()
    plt.show()

    print(f"Variance residuals: {residuals.var()}\n")

def filter_data_by_range(df, column, min_val, max_val):
    if column not in df.columns:
        print(f"Kolumna '{column}' nie istnieje w zbiorze danych.\n")
        return df

    initial_count = df.shape[0]
    condition = (df[column] >= min_val) & (df[column] <= max_val)
    filtered_df = df[condition]
    removed = initial_count - filtered_df.shape[0]
    print(f"Usunięto {removed} wartości odstających z kolumny '{column}' w zakresie [{min_val}, {max_val}]")
    print(f"Rozmiar danych: {filtered_df.shape}\n")
    return filtered_df

def analyze_train_dataset():
    train_df = pd.read_csv('train.csv')


    train_set, test_set = train_test_split(train_df, test_size=0.2, random_state=42)
    #print(f"Zbiór treningowy: {train_set.shape}")
    #print(f"Zbiór testowy: {test_set.shape}\n")

    missing_train = train_set.isnull().sum().sort_values(ascending=False)
    missing_percent_train = (train_set.isnull().sum() / train_set.shape[0]).sort_values(ascending=False)
    missing_info_train = pd.concat([missing_train, missing_percent_train], axis=1, keys=['Total', 'Percentage'])
    #print("Brakujące dane w zbiorze treningowym:")
    #print(missing_info_train.head(20), "\n")

    columns_to_drop = missing_info_train[missing_info_train['Total'] > 1].index
    train_set = train_set.drop(columns=columns_to_drop, axis=1)
    train_set = train_set.drop(train_set[train_set['Electrical'].isnull()].index)
    #print(f"Zbiór po usunięciu brakujących wartości: {train_set.shape}\n")

    test_set = test_set[train_set.columns]

    outlier_thresholds = {
        '1stFlrSF': {'upper': 2500},
        'GarageArea': {'upper': 1250},
        'GrLivArea': {'upper': 4000},
        'TotalBsmtSF': {'upper': 3000}

    }

    for feature, limits in outlier_thresholds.items():
        if feature in train_set.columns:
            train_set = exclude_outliers(train_set, feature, limits['upper'])
            test_set = exclude_outliers(test_set, feature, limits['upper'])

    regression_models = {
        'Model_GrLivArea': 'SalePrice ~ GrLivArea',
        'Model_OverallQual': 'SalePrice ~ OverallQual',
        'Model_Multiple': 'SalePrice ~ GrLivArea + OverallQual + TotalBsmtSF'
    }

    fitted_models = {}

    for model_name, formula in regression_models.items():
        print('-' * 75)
        print(f'{model_name}: {formula}')
        model = ols(formula, data=train_set).fit()
        print(model.summary(), "\n")
        fitted_models[model_name] = model
        main_predictor = formula.split('~')[1].strip().split('+')[0].strip()
        plot(train_set, model, main_predictor, 'SalePrice')

    print('-' * 75)
    print('Model Random Forest')
    rf_features = ['YearBuilt', 'OverallQual', 'GrLivArea', 'GarageCars']
    rf_features_present = [feature for feature in rf_features if feature in train_set.columns]
    if len(rf_features_present) == len(rf_features):
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(train_set[rf_features], train_set['SalePrice'])
        test_set['Pred_RF'] = rf_model.predict(test_set[rf_features])
    else:
        missing_features = set(rf_features) - set(rf_features_present)
        print(f"Brakujące cechy dla modelu Random Forest: {missing_features}\n")

    for model_name, model in fitted_models.items():
        test_set[f'Pred_{model_name}'] = model.predict(test_set)

    r2_scores = {}
    for model_name in regression_models.keys():
        pred_col = f'Pred_{model_name}'
        if pred_col in test_set.columns:
            r2 = r2_score(test_set['SalePrice'], test_set[pred_col])
            r2_scores[model_name] = r2
            print(f'R2 dla {model_name}: {r2:.4f}')

    if 'Pred_RF' in test_set.columns:
        r2_rf = r2_score(test_set['SalePrice'], test_set['Pred_RF'])
        r2_scores['RandomForest'] = r2_rf
        print(f'R2 dla Modelu Random Forest: {r2_rf:.4f}\n')
    print("""
        Model liniowy wykorzystujący jedynie GrLivArea wyjaśnia około 52,66% zmienności w cenach sprzedaży SalePrice.
        Zmienna OverallQual tłumaczy 66,82% zmienności cen, co podkreśla jej kluczową rolę jako predyktora w kontekście wartości nieruchomości.
        SalePrice ~ GrLivArea + OverallQual + TotalBsmtSF wyjaśnia 79,81% zmienności cen sprzedaży. Wartość R² oraz mniejsza wariancja reszt w porównaniu do pozostałych modeli liniowych wskazują, że uwzględnienie dodatkowych cech pozwala lepiej uchwycić złożoność danych.
        Model Random Forest uzyskał najwyższy współczynnik determinacji R², równy 0.8450, co wskazuje na jego najwyższą skuteczność w wyjaśnianiu zmienności zmiennej zależnej SalePrice dla podanego zbioru.
        Wyniki te potwierdzają, że model Random Forest, dzięki swojej zdolności do modelowania nieliniowych i złożonych relacji jest najefektywniejszy w predykcji cen nieruchomości.
    """)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(r2_scores.keys()), y=list(r2_scores.values()))
    plt.ylabel('R2 Score')
    plt.title('Porównanie R2 dla różnych modeli na zbiorze testowym')
    plt.ylim(0, 1)
    plt.show()

def analyze_deflection_dataset():
    deflection_df = pd.read_csv('deflection.csv', delimiter=';')
    #print(deflection_df.head(), "\n")

    print('-' * 75)
    print('Model Liniowy: Deflection ~ Load -1')
    linear_model = ols('Deflection ~ Load -1', data=deflection_df).fit()
    print(linear_model.summary(), "\n")
    plot(deflection_df, linear_model, 'Load', 'Deflection')

    deflection_df['Load_Squared'] = deflection_df['Load'] ** 2
    print('-' * 75)
    print('Model Deflection ~ Load + Load_Squared')
    quadratic_model = ols('Deflection ~ Load + Load_Squared', data=deflection_df).fit()
    print(quadratic_model.summary(), "\n")
    plot(deflection_df, quadratic_model, 'Load', 'Deflection')
    print('-' * 75)

def main():
    analyze_train_dataset()
    print("\n" + "=" * 100 + "\n")
    analyze_deflection_dataset()

if __name__ == "__main__":
    main()



