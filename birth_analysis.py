import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3

def load_data_concat(data_path: str) -> pd.DataFrame:

    data_dir = Path(data_path)

    def file_reader():
        for file in data_dir.glob('yob*.txt'):
            yield pd.read_csv(
                file,
                header=None,
                names=['Name', 'Gender', 'Count'],
                dtype={'Name': 'str', 'Gender': 'category', 'Count': 'int32'}
            ).assign(Year=int(file.stem[3:]))

    return pd.concat(file_reader(), ignore_index=True)

# Funkcja do zliczenia unikalnych imion
def count_unique_names(df: pd.DataFrame) -> int:
    unique_count = df['Name'].nunique()
    print(f"Całkowita liczba unikalnych imion: {unique_count}")
    return unique_count

# Funkcja do zliczenia unikalnych imion według płci
def count_unique_names_by_gender(df: pd.DataFrame) -> pd.Series:
    unique_names_gender = df.groupby('Gender')['Name'].nunique()
    print(f"Unikalne męskie imiona: {unique_names_gender.get('M', 0)}")
    print(f"Unikalne żeńskie imiona: {unique_names_gender.get('F', 0)}")
    return unique_names_gender

# Funkcja do obliczenia częstotliwości
def calculate_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    df['Total'] = df.groupby(['Year', 'Gender'])['Count'].transform('sum')
    df['Frequency'] = df['Count'] / df['Total']
    return df

# Funkcja do wykreślenia statystyk urodzeń
def plot_birth_statistics(df: pd.DataFrame):

    births_by_year_gender = df.groupby(['Year', 'Gender'])['Count'].sum().unstack(fill_value=0)
    births_by_year_gender['Total'] = births_by_year_gender.sum(axis=1)
    births_by_year_gender['F_M_Ratio'] = births_by_year_gender.get('F', 0) / births_by_year_gender.get('M', 1)
    min_ratio_year = births_by_year_gender['F_M_Ratio'].idxmin()
    max_ratio_year = births_by_year_gender['F_M_Ratio'].idxmax()
    min_ratio = births_by_year_gender['F_M_Ratio'].min()
    max_ratio = births_by_year_gender['F_M_Ratio'].max()

    print(f"Najmniejszy stosunek F/M: {min_ratio:.4f} w roku {min_ratio_year}")
    print(f"Największy stosunek F/M: {max_ratio:.4f} w roku {max_ratio_year}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    births_by_year_gender['Total'].plot(ax=ax1, color='purple')
    ax1.set_title('Całkowita liczba urodzeń w USA na rok')
    ax1.set_ylabel('Liczba urodzeń')

    births_by_year_gender['F_M_Ratio'].plot(ax=ax2, color='green')
    ax2.set_title('Stosunek liczby urodzeń dziewczynek do chłopców w USA')
    ax2.set_ylabel('Stosunek F/M')
    ax2.set_xlabel('Rok')
    ax2.scatter([min_ratio_year, max_ratio_year], [min_ratio, max_ratio], color='red', zorder=5)
    ax2.annotate(f"Min: {min_ratio:.4f}\n({min_ratio_year})",
                 xy=(min_ratio_year, min_ratio),
                 xytext=(min_ratio_year, min_ratio * 0.95),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)
    ax2.annotate(f"Max: {max_ratio:.4f}\n({max_ratio_year})",
                 xy=(max_ratio_year, max_ratio),
                 xytext=(max_ratio_year, max_ratio * 1.05),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)

    plt.tight_layout()
    #plt.show()



def determine_top1000_names(df: pd.DataFrame) -> dict:
    weighted_freq = df.groupby(['Gender', 'Name'], observed=False)['Frequency'].sum().reset_index()

    # Sortowanie i wybieranie Top 1000 dla każdej płci
    top1000 = weighted_freq.sort_values(['Gender', 'Frequency'], ascending=[True, False]).groupby('Gender', observed=False).head(1000)
    return {gender: group.sort_values(by='Frequency', ascending=False) for gender, group in top1000.groupby('Gender', observed=False)}


def plot_selected_names_trend(df: pd.DataFrame, top1000: dict):
    male_name = 'John'
    #female_name = 'Mary'
    most_popular_female_name = df[df['Gender'] == 'F'].loc[df[df['Gender'] == 'F']['Frequency'].idxmax(), 'Name']
    female_name = most_popular_female_name

    selected_names = [male_name, female_name]

    selected_data = df[df['Name'].isin(selected_names)]
    pivot_counts = selected_data.pivot_table(index='Year', columns=['Name', 'Gender'], values='Count', aggfunc='sum').fillna(0)
    pivot_freq = selected_data.pivot_table(index='Year', columns=['Name', 'Gender'], values='Frequency', aggfunc='sum').fillna(0)

    pivot_counts.columns = ['_'.join(col).strip() for col in pivot_counts.columns.values]
    pivot_freq.columns = ['_'.join(col).strip() for col in pivot_freq.columns.values]

    specific_years = [1934, 1980, 2022]
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color_map = {'M': 'blue', 'F': 'pink'}
    for name in selected_names:
        gender = 'M' if name == male_name else 'F'
        col_name = f"{name}_{gender}"
        ax1.plot(pivot_counts.index, pivot_counts[col_name], label=f'Liczba {name} ({gender})', color=color_map[gender])
        for year in specific_years:
            if year in pivot_counts.index:
                count = pivot_counts.loc[year, col_name]
                ax1.annotate(f"{int(count)}", (year, count), textcoords="offset points", xytext=(0, 10), ha='center')

    ax1.set_xlabel('Rok')
    ax1.set_ylabel('Liczba nadanych imion')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    for name in selected_names:
        gender = 'M' if name == male_name else 'F'
        freq_col_name = f"{name}_{gender}"
        ax2.plot(pivot_freq.index, pivot_freq[freq_col_name], label=f'Popularność {name} ({gender})', linestyle='--', color=color_map[gender])

    ax2.set_ylabel('Popularność imienia')
    ax2.legend(loc='upper right')
    plt.title('Trendy popularności imion: John i Mary')
    #plt.show()

    for name in selected_names:
        gender = 'M' if name == male_name else 'F'
        col_name = f"{name}_{gender}"
        for year in specific_years:
            if year in pivot_counts.index:
                count = pivot_counts.loc[year, col_name]
                print(f"Liczba {name} ({gender}) w roku {year}: {int(count)}")
# Funkcja do wykreślenia różnorodności imion z indeksem
def plot_name_diversity_with_index(df: pd.DataFrame, top1000: dict):

    top1000_names = pd.concat([
        top1000['M'][['Name']].assign(Gender='M'),
        top1000['F'][['Name']].assign(Gender='F')
    ]).set_index(['Name', 'Gender'])

    df = df.set_index(['Name', 'Gender'])
    df['in_top'] = df.index.isin(top1000_names.index)
    df.reset_index(inplace=True)

    # Grupowanie danych i obliczenie procentu imion z Top 1000
    grouped = df.groupby(['Year', 'Gender']).apply(
        lambda x: x.loc[x['in_top'], 'Count'].sum() / x['Count'].sum()
    ).unstack()
    grouped.columns = ['Mężczyźni', 'Kobiety']

    # Obliczenie różnic między męskimi a żeńskimi imionami
    grouped['Różnica'] = abs(grouped['Mężczyźni'] - grouped['Kobiety'])

    # Znalezienie lat z największą i najmniejszą różnicą
    max_diff_year = grouped['Różnica'].idxmax()
    max_diff_value = grouped['Różnica'].max()

    min_diff_year = grouped['Różnica'].idxmin()
    min_diff_value = grouped['Różnica'].min()

    print(f"Największa różnica w różnorodności między męskimi a żeńskimi imionami wystąpiła w roku {max_diff_year} i wynosiła {max_diff_value:.4f}")
    print(f"Najmniejsza różnica w różnorodności między męskimi a żeńskimi imionami wystąpiła w roku {min_diff_year} i wynosiła {min_diff_value:.4f}")

    fig, ax = plt.subplots(figsize=(15, 7))
    grouped[['Mężczyźni', 'Kobiety']].plot(ax=ax, color=['blue', 'pink'])
    ax.axvline(max_diff_year, color='green', linestyle='--', label=f'Największa różnica: {max_diff_year}')
    ax.axvline(min_diff_year, color='red', linestyle='--', label=f'Najmniejsza różnica: {min_diff_year}')
    ax.annotate(f"{max_diff_value:.2f}", xy=(max_diff_year, (grouped.loc[max_diff_year, 'Mężczyźni'] + grouped.loc[max_diff_year, 'Kobiety']) / 2), xytext=(max_diff_year, 0.6))
    ax.annotate(f"{min_diff_value:.2f}", xy=(min_diff_year, (grouped.loc[min_diff_year, 'Mężczyźni'] + grouped.loc[min_diff_year, 'Kobiety']) / 2), xytext=(min_diff_year, 0.6))
    ax.set_title('Procent imion z Top 1000 w danym roku (podział na płeć)')
    ax.set_ylabel('Procent imion z Top 1000')
    ax.set_xlabel('Rok')
    ax.legend()
    print("\n Po 1984 następuje wyraźny spadek udziału imion z top 1000 wsród chłopców, co wskazuje na odchodzenie od tradycyjnych wzorców ")
    print("\n W XXI wieku zwłaszcza po 2015 roku następuje wyraźne zmniejszczenie się udziału imion z top 1000 wsród dziewczynek i chłopców (odpowiednio około 62% i 78%, co może wynikać z pogłebienia globalizacji")
    #plt.show()

# Funkcja do analizy rozkładu ostatnich liter
def analyze_last_letter_distribution(df: pd.DataFrame):
    # Dodanie kolumny z ostatnią literą imienia
    df['Last_Letter'] = df['Name'].str[-1]

    # Filtrowanie tylko męskich imion
    male_df = df[df['Gender'] == 'M']

    # Grupowanie danych i agregacja dla ostatnich liter
    last_letter_ct = male_df.pivot_table(
        index='Year', columns='Last_Letter', values='Count', aggfunc='sum', fill_value=0
    )
    normalized_last_letters = last_letter_ct.div(last_letter_ct.sum(axis=1), axis=0)

    # Wyodrębnienie danych dla lat 1910, 1970, 2023
    selected_years = normalized_last_letters.loc[[1910, 1970, 2023]]
    selected_years.T.plot(kind='bar', figsize=(15, 7), alpha=0.8)
    plt.title('Rozkład ostatnich liter męskich imion w latach 1910, 1970, 2023')
    plt.ylabel('Procent')
    plt.xlabel('Ostatnia litera')
    plt.legend(title='Rok')
    #plt.show()
    print("\n  W 1910 roku rozkład ostatnich liter był bardziej równomierny, z wyraźnym udziałem liter takich jak „e”, „s”, „d” i „r”. W 2023 roku zauważalna jest koncentracja na literach „o” i „n”,")
    print("\n W 2023 roku literą dominującą stało się „o”, które osiągnęło udział ponad 30%. To istotna zmiana w rozkładzie może wynikać z globalizacji i rosnących wpływów różnych grup etnicznych w populacji USA (np. wśród społeczności latynoskich). Spadek znaczenia liter „e” czy „d” wskazuje na stopniowe odchodzenie od tradycyjnych imion anglosaskich")
    # Obliczenie różnic w popularności liter między 1910 a 2023
    differences = selected_years.loc[2023] - selected_years.loc[1910]
    max_increase = differences.idxmax()
    max_decrease = differences.idxmin()

    print(f"Największy wzrost popularności: litera '{max_increase}' ({differences[max_increase]:.4f})")
    print(f"Największy spadek popularności: litera '{max_decrease}' ({differences[max_decrease]:.4f})")

    # Wybranie 3 liter o największej zmianie
    top_changes = differences.abs().nlargest(3).index


    plt.figure(figsize=(15, 7))
    for letter in top_changes:
        plt.plot(normalized_last_letters.index, normalized_last_letters[letter], label=f'Ostatnia litera: {letter}')

    plt.title('Trendy popularności wybranych ostatnich liter na przestrzeni lat')
    plt.xlabel('Rok')
    plt.ylabel('Procent')
    plt.legend()
    #plt.show()

# Funkcja do obliczenia proporcji płci imion w okresie
def calculate_name_gender_ratios_in_period(df: pd.DataFrame, top1000: dict, start_year: int, end_year: int):

    # Filtrowanie danych dla podanego okresu
    period_df = df[df['Year'].between(start_year, end_year)]

    # Filtrowanie imion w Top 1000 dla obu płci
    top_male_names = top1000['M'][['Name']].assign(Gender='M')
    top_female_names = top1000['F'][['Name']].assign(Gender='F')
    top_names_both = pd.merge(top_male_names, top_female_names, on='Name', how='inner')['Name'].unique()

    # Filtrowanie imion, które są w Top 1000 dla obu płci
    period_df = period_df[period_df['Name'].isin(top_names_both)]

    grouped_df = period_df.groupby(['Name', 'Gender'])['Count'].sum().unstack(fill_value=0)
    grouped_df.columns = ['Count_F', 'Count_M']

    grouped_df['p_m'] = grouped_df['Count_M'] / (grouped_df['Count_M'] + grouped_df['Count_F'])
    grouped_df['p_f'] = 1 - grouped_df['p_m']

    return grouped_df

# Funkcja do znalezienia znaczących zmian płci imion
def find_significant_gender_shifts(df: pd.DataFrame, top1000: dict):

    # Obliczenie proporcji płci dla dwóch okresów
    ratios_pre_1920 = calculate_name_gender_ratios_in_period(df, top1000, start_year=1880, end_year=1920)
    ratios_post_2000 = calculate_name_gender_ratios_in_period(df, top1000, start_year=2000, end_year=2023)

    # Łączenie danych w celu obliczenia zmian
    merged_ratios = pd.merge(
        ratios_pre_1920[['p_m']],
        ratios_post_2000[['p_m']],
        left_index=True,
        right_index=True,
        suffixes=('_pre', '_post')
    )

    # Obliczenie zmiany w proporcji męskiej
    merged_ratios['change'] = merged_ratios['p_m_post'] - merged_ratios['p_m_pre']

    # Znalezienie imienia z największym przesunięciem z męskiego na żeńskie
    max_m2f_name = merged_ratios['change'].idxmin()
    max_m2f_value = merged_ratios.loc[max_m2f_name, 'change']

    # Znalezienie imienia z największym przesunięciem z żeńskiego na męskie
    max_f2m_name = merged_ratios['change'].idxmax()
    max_f2m_value = merged_ratios.loc[max_f2m_name, 'change']

    print(f"Imię zmieniło konotację z męskiej na żeńską: {max_m2f_name} (Zmiana: {max_m2f_value:.4f})")
    print(f"Imię zmieniło konotację z żeńskiej na męską: {max_f2m_name} (Zmiana: {max_f2m_value:.4f})")

    return max_m2f_name, max_f2m_name

# Funkcja do wykreślenia trendów konotacji płciowej
def plot_gender_connotation_trends(df: pd.DataFrame, names: list):
    # Filtrowanie danych dla wybranych imion
    trend_data = df[df['Name'].isin(names)]

    # Grupowanie po roku, imieniu i płci oraz sumowanie liczby
    trend_grouped = trend_data.groupby(['Year', 'Name', 'Gender'])['Count'].sum().unstack('Gender', fill_value=0)


    for gender in ['M', 'F']:
        if gender not in trend_grouped.columns:
            trend_grouped[gender] = 0

    # Obliczanie całkowitej liczby i proporcji
    trend_grouped['Total'] = trend_grouped['M'] + trend_grouped['F']

    trend_grouped['p_m'] = trend_grouped.apply(
        lambda row: row['M'] / row['Total'] if row['Total'] > 0 else 0, axis=1
    )
    trend_grouped['p_f'] = 1 - trend_grouped['p_m']

    # Wykres trendów konotacji płciowej
    plt.figure(figsize=(15, 7))
    for name in names:
        if name in trend_grouped.index.get_level_values('Name'):
            name_data = trend_grouped.xs(name, level='Name')
            plt.plot(name_data.index, name_data['p_m'], label=f"{name} (mężczyźni)", linestyle='-', marker='o')
            plt.plot(name_data.index, name_data['p_f'], label=f"{name} (kobiety)", linestyle='--', marker='x')

    plt.title('Trendy konotacji płciowej dla wybranych imion (Top 1000)')
    plt.xlabel('Rok')
    plt.ylabel('Proporcja')
    plt.legend()
    plt.grid()
    #plt.show()


# Sekcja 2: Analiza Imion w Polsce

# Wczytanie Danych z Polski z Bazy SQLite
def load_polish_data(database_path: str) -> pd.DataFrame:
    with sqlite3.connect(database_path) as conn:
        males = pd.read_sql("SELECT Imię AS Name, Liczba AS Count, Rok AS Year FROM males", conn)
        females = pd.read_sql("SELECT Imię AS Name, Liczba AS Count, Rok AS Year FROM females", conn)
    males['Gender'] = 'Male'
    females['Gender'] = 'Female'
    return pd.concat([males, females], ignore_index=True)

# Obliczanie Różnorodności Top 200 Imion w Polsce
def calculate_top200_diversity_polish(df: pd.DataFrame, start_year=2000, end_year=2023, top_n: int=100) -> pd.DataFrame:
    filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    totals = filtered.groupby(['Year', 'Gender'])['Count'].sum()
    filtered['Rank'] = filtered.groupby(['Year', 'Gender'])['Count'].rank(ascending=False, method='first')
    top200 = filtered[filtered['Rank'] <= top_n].groupby(['Year', 'Gender'])['Count'].sum()
    diversity = (top200 / totals).reset_index(name='Top200_Percentage')
    return diversity

# Zadanie 3: Znalezienie Imion Neutralnych Płciowo w Polsce
def find_gender_neutral_names_polish(df: pd.DataFrame, max_difference=0.5) -> pd.DataFrame:
    gender_sums = df.groupby(['Name', 'Gender'])['Count'].sum().unstack(fill_value=0)
    gender_sums['Total'] = gender_sums['Male'] + gender_sums['Female']
    gender_sums['Proportion_Male'] = gender_sums['Male'] / gender_sums['Total']
    gender_sums['Proportion_Female'] = 1 - gender_sums['Proportion_Male']
    gender_sums['Ratio_Difference'] = abs(gender_sums['Proportion_Male'] - gender_sums['Proportion_Female'])
    return gender_sums[gender_sums['Ratio_Difference'] <= max_difference].sort_values(by='Total', ascending=False)

# Zadanie 2: Wykres Różnorodności Imion w Polsce
def plot_name_diversity(data: pd.DataFrame, top_n: int, start_year: int, end_year: int, dataset_name: str):

        filtered = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]

        # Obliczenie całkowitej liczby urodzeń w każdym roku i płci
        totals = filtered.groupby(['Year', 'Gender'])['Count'].sum()
        filtered['Rank'] = filtered.groupby(['Year', 'Gender'])['Count'].rank(ascending=False, method='first')

        # Filtrowanie danych dla Top N imion
        top_n_data = filtered[filtered['Rank'] <= top_n].groupby(['Year', 'Gender'])['Count'].sum()
        diversity = (top_n_data / totals).reset_index(name='TopN_Percentage')
        plt.figure(figsize=(14, 7))
        for gender in diversity['Gender'].unique():
            gender_data = diversity[diversity['Gender'] == gender]
            plt.plot(gender_data['Year'], gender_data['TopN_Percentage'] * 100, label=f"{gender} ({dataset_name})")
        plt.title(f"Różnorodność imion w Top {top_n} ({dataset_name}) w latach {start_year}-{end_year}")
        plt.xlabel("Rok")
        plt.ylabel(f"Procent imion w Top {top_n} (%)")
        plt.axvline(2000, color='red', linestyle='--', label='Rok 2000')
        plt.axvline(2013, color='green', linestyle='--', label='Rok 2013')
        plt.axvline(end_year, color='blue', linestyle='--', label=f'Rok {end_year}')
        plt.legend(title="Płeć")
        plt.grid()

        #plt.show()
        print( "\n ODP: Spadek udziału imion z top1000 jest statystycznym odwierciedleniem zmian społecznych, takich jak migracje, globalizacja, wpływu popkultury. Po 2020 roku na wykresie róźnorodności imion widoczny jest wyraźny spadek, niemal pionowy spadek udziału imion z top200 w Polsce,zarówno dla chłopców jak i dziewczynek.")
        print("\n Po 2 wojnie światowej udział imion z top100 zaczął się zmniejszać. Podobnie po 1980 roku  imion męskich wzrasta znacznie szybciej, co może wskazywać na stopniowe odchodzenie od konserwatywnych wzorców.")
        print("\n Po 2013 roku nastąpiła wyrażna zmiana, była ona wynikiem zmian kryterium raportowania danych statystycznych. Wprowadzenie w statystykach progu 2 nadań zamiast wcześniejszego minimum 5 nadań umożliwiło rejestrację większej liczby unikalnych imion, co zwiększyło widoczność imion spoza (Top 200).")


def main():

    #data_url = 'https://www.ssa.gov/oact/babynames/names.zip'

    data_directory = "./data"
    df = load_data_concat(data_directory)
    count_unique_names(df)
    count_unique_names_by_gender(df)
    df = calculate_frequencies(df)
    plot_birth_statistics(df)

    top1000 = determine_top1000_names(df)

    specific_years = [1934, 1980, 2022]
    plot_selected_names_trend(df, specific_years)

    plot_name_diversity_with_index(df, top1000)

    analyze_last_letter_distribution(df)

    max_m2f_name, max_f2m_name = find_significant_gender_shifts(df, top1000)
    plot_gender_connotation_trends(df, names=[max_m2f_name, max_f2m_name])

    polish_database_path = './data/names_pl_2000-23.sqlite'
    polish_df = load_polish_data(polish_database_path)

    polish_diversity = calculate_top200_diversity_polish(polish_df, start_year=2000, end_year=2023, top_n=100)

    plot_name_diversity(data=polish_df, top_n=200, start_year=2000, end_year=2023, dataset_name="Polska")
    #plot_name_diversity(data=df, top_n=1000, start_year=1880, end_year=2023, dataset_name="USA")

    polish_gender_neutral = find_gender_neutral_names_polish(polish_df)
    print(polish_gender_neutral.head(2))
    plt.show()

if __name__ == "__main__":
    main()
