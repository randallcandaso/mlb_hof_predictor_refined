"""

The program aims to predict Hall of Fame caliber players in the current crop of active MLB talent.
First, projected stats are calculated for each active player, with a minimum of four seasons,
up until a baseline age that is determined fit for retirement. Then, all Hall of Fame inductees, a select
number of non-hall-of-fame retired players, and the players used in the projection are combined into
one singular dataframe. Using the dataframe and each individual's 'status', a prediction model is created
and fitted for the use of predicting who among current active players has the potential to be inducted
into the Hall of Fame upon their retirement.

"""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import randint
import pandas as pd
import numpy as np
import sqlite3
import math


def read_and_encode_1(file, check=False):
    """

    Function reads in a file name and converts it into a dataframe. Files read into this
     function are specifically encoded in ISO-8859-1.

    :param file: desired file name
    :param check: whether a file needs its index fixed
    :return: file as a dataframe
    """
    pd.options.mode.chained_assignment = None
    blank = pd.read_csv(file, encoding='ISO-8859-1')
    blank = blank.replace({'Â': '', 'Ã©': 'é', 'Ã': '', '±': 'ñ'}, regex=True)
    if check == True:
        blank.set_index(blank.columns[0], inplace=True)
    return blank


def read_and_encode_2(file):
    """

    Similar to read_and_encode_1, a file name is read in and formatted into a dataframe.
    Files read into this function are specifically encoded in utf-8.

    :param file: desired file name
    :return: file as a dataframe
    """
    blank = pd.read_csv(file, encoding='utf-8')
    return blank


def fix_age(players_hof):
    """

    For the specific dataframe inputted, the 'Age' column is properly formatted using the
    fix_age_helper function. Additionally, a value is calculated from this column to use
    as a baseline for later computational work in the program.

    :param players_hof: dataframe containing hall of fame players
    :return: the newly formatted players_hof df, a calculated value for later use
    """
    players_hof = players_hof.fillna(0)
    players_hof['Age'] = players_hof['Age'].apply(fix_age_helper)
    hof_age_mean = players_hof['Age'].mean() - 2
    hof_age_mean = math.ceil(hof_age_mean)
    players_hof = format_hof(players_hof)
    return players_hof, hof_age_mean


def fix_age_helper(age):
    """

    Helper function for formatting an age value from string to int, if needed.

    :param age: inputted value from fix_age function
    :return: the now integer value
    """
    if '-' in age:
        return int(age.split('-')[1])
    else:
        return int(age)


def format_hof(players_hof):
    """

    The players_hof dataframe columns are re-ordered for easier use later in the program.

    :param players_hof: dataframe containing hall of fame players
    :return: the re-ordered dataframe
    """
    order_hof_cols = ['Player', 'WAR', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR',
                      'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
                      'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser', 'Rbaser + Rdp',
                      'Rfield', 'BA', 'OBP', 'SLG', 'OPS']
    players_hof = players_hof[order_hof_cols]
    return players_hof


def log_transform(X):
    """

    Function used to calculate a log value for a prediction model.

    :param X: base value for calculation
    :return: newly calculated value
    """
    X = X.clip(min=1e-8)
    return np.log(X)


def calc_bat_avg(hits, at_bats):
    """

    Function calculates a specific player's batting average.

    :param hits: number of hits
    :param at_bats: number of at bats
    :return: the batting average represented as a float
    """
    if at_bats == 0:
        return 0.0
    return round((hits / at_bats), 3)


def calc_obp(hits, walks, hbp, at_bats, sf):
    """

    Function calculates a specific player's on base percentage.

    :param hits: number of hits
    :param walks: number of walks
    :param hbp: number of hit by pitches
    :param at_bats: number of at bats
    :param sf: number of sacrificial fly balls
    :return: the on base percentage represented as a float
    """
    first = hits + walks + hbp
    second = at_bats + hbp + sf
    if second == 0:
        return 0.0
    return round((first / second), 3)


def calc_slug(singles, doubles, triples, hrs, at_bats):
    """

    Function calculates a specific player's slugging percentage.

    :param singles: number of singles hit
    :param doubles: number of doubles hit
    :param triples: number of triples hit
    :param hrs: number of home runs hit
    :param at_bats: number of at bats
    :return: the slugging percentage represented as a float
    """
    if at_bats == 0:
        return 0.0
    doubles = doubles * 2
    triples = triples * 3
    home_bs = hrs * 4
    return round(((singles + doubles + triples + home_bs) / at_bats), 3)


def calc_ops(obp, slug):
    """

    Function calculates a specific player's OPS value.

    :param obp: player's on base percentage
    :param slug: player's slugging percentage
    :return: OPS value represented as a float
    """
    return round((obp + slug), 3)


def check_active(year, glover, act_temp):
    """

    Checks the active player dataframe to see if 'glover' is present.

    :param year: the year of the 'glover' season
    :param glover: list containing 'glover' name and 'glover' team
    :param act_temp: active player dataframe
    :return: the full player name if matched
    """
    for i in act_temp.index:
        player = act_temp.loc[i]
        name = player['Player'].split()
        if name[1] == glover[0] and int(player['Season']) == int(year):
            if player['Team'] == glover[1]:
                return player['Player']
    return None


def check_hof(year, glover, hof_temp):
    """

    Checks the hall of fame player dataframe to see if 'glover' is present.

    :param year: the year of the 'glover' season
    :param glover: list containing 'glover' name and 'glover' team
    :param hof_temp: hall of fame dataframe
    :return: the full player name if matched
    """
    for i in hof_temp.index:
        player = hof_temp.loc[i]
        name = player['Player'].split()
        if name[1] == glover[0] and glover[1] in player['Team'].split(','):
            if int(player['From']) <= int(year) <= int(player['To']):
                return player['Player']
    return None


def check_retired(year, glover, ret_temp):
    """

    Checks the retired player dataframe to see if 'glover' is present.

    :param year: the year of the 'glover' season
    :param glover: list containing 'glover' name and 'glover' team
    :param ret_temp: hall of fame dataframe
    :return: the full player name if matched
    """
    for i in ret_temp.index:
        player = ret_temp.loc[i]
        name = player['Player'].split()
        if name[1] == glover[0] and glover[1] in player['Team'].split(','):
            if int(player['From']) <= int(year) <= int(player['To']):
                return player['Player']
    return None


def setup_award(merged_data, award, df_1, df_2, hof_temp, act_temp, ret_temp):
    """

    Function utilizes setup functions and inputted dataframes to calculate the number of
    gold gloves and silver slugger awards each player, contained in merged_data, has
    respectively.

    :param merged_data: dataframe containing a mixture of mlb players
    :param award: specific award desired
    :param df_1: an American League Dataframe
    :param df_2: a National League Dataframe
    :param hof_temp: hall of fame dataframe
    :param act_temp: active player dataframe
    :param ret_temp: retired player dataframe
    """
    merged_data[award] = 0
    al_temp = setup_helper(df_1)
    nl_temp = setup_helper(df_2)

    gg_ss_checks(al_temp, merged_data, award, act_temp, ret_temp, hof_temp)
    gg_ss_checks(nl_temp, merged_data, award, act_temp, ret_temp, hof_temp)


def setup_helper(df):
    """

    Helper function used to properly format an inputted award dataframe for use in setup_award.

    :param df: award dataframe
    :return: reformatted award dataframe
    """
    df = df.copy()
    df.set_index(df.columns[0], inplace=True)
    team = 'Team'
    if team in df.columns:
        df.drop('Team', axis=1, inplace=True)
    return df


def gg_ss_checks(league, whole, col, act_temp, ret_temp, hof_temp):
    """

    Function transverses an award dataframe and matches each award winner
    to, if present, a player in the main dataframe used in the program.

    :param league: specific league award dataframe
    :param whole: main dataframe used in the program
    :param col: specific award's string representation
    :param act_temp: active player dataframe
    :param ret_temp: retired player dataframe
    :param hof_temp: hall of fame dataframe
    """
    for i in league.index:
        for j in league.columns:
            if pd.isna(league.loc[i, j]):
                continue
            glover = league.loc[i, j].split(',')
            glover = [item.strip('\xa0') for item in glover]
            year = i.split()
            check_1 = check_active(year[0], glover, act_temp)
            check_2 = check_hof(year[0], glover, hof_temp)
            check_3 = check_retired(year[0], glover, ret_temp)
            if check_1:
                whole.loc[whole['Player'] == check_1, col] += 1
            if check_2:
                whole.loc[whole['Player'] == check_2, col] += 1
            if check_3:
                whole.loc[whole['Player'] == check_3, col] += 1


def simple_awards(df, award_df, award, count=False):
    """

    For other specific awards, the function is used for similar functionality as gg_ss_checks,
    but needing much simpler implementation.

    :param df: main dataframe used in the program
    :param award_df: specific award dataframe
    :param award: specific award's string representation
    :param count: if an award requires a counting number
    """
    df[award] = 0
    if count == False:
        df[award] = False
    award_df_2 = award_df.copy()

    for i in award_df_2.index:
        player = award_df_2.iloc[i, 2]
        for j in df.index:
            temp = df.loc[j, 'Player']
            if temp == player:
                if count == False:
                    df.loc[j, award] = True
                else:
                    df.loc[j, award] += 1


def ps_awards(merged_data, ps_mvps):
    """

    Function is implemented for the specific format of the post season award
    dataframe. Its functionality is similar to the previous award-checking
    functions.

    :param merged_data: main dataframe used in the program
    :param ps_mvps: post season award dataframe
    """
    merged_data['PS_MVPs'] = 0
    ps_2 = ps_mvps.copy()
    ps_2.set_index(ps_2.columns[0], inplace=True)

    for i in ps_2.index:
        for j in ps_2.columns:
            if pd.isna(ps_2.loc[i, j]):
                continue
            player = ps_2.loc[i, j]
            if ',' in player:
                blank = player.split()
                player_1 = blank[0]
                player_2 = blank[1]
                blank = [player_1, player_2]
                for count in blank:
                    for x in merged_data.index:
                        temp = merged_data.loc[x, 'Player']
                        if temp == count:
                            merged_data.loc[x, 'PS_MVPs'] += 1
            else:
                for y in merged_data.index:
                    temp = merged_data.loc[y, 'Player']
                    if temp == player:
                        merged_data.loc[y, 'PS_MVPs'] += 1


def bt_create(merged_data, bat_titles):
    """

    First the function fixes the string values in the award dataframe. It then uses
    the simple_awards function to add the specific players who have won a batting
    title.

    :param merged_data: main dataframe used in the program
    :param bat_titles: batting award dataframe
    :return:
    """
    merged_data['Bat_Titles'] = 0
    bt_2 = bat_titles.copy()
    bt_2['Batting Champ'] = bt_2['Batting Champ'].str.replace('\xa0', ' ')

    simple_awards(merged_data, bt_2, 'Bat_Titles', count=True)


def as_create(merged_data, all_stars):
    """

    The all stars dataframe is reformatted for proper use. Then, the simple_awards
    function is used to count for the specific players who have won been voted as an all
    star in the past.

    :param merged_data: main dataframe used in the program
    :param all_stars: all stars dataframe
    """
    as_2 = all_stars.copy()
    as_2.set_index(as_2.columns[0], inplace=True)
    as_2 = as_2.reset_index(drop=True)

    simple_awards(merged_data, as_2, 'All_Stars', count=True)


def format_after_awards(merged_data):
    """

    Function re-orders some of the award columns for easier use of the df
    later in the program.

    :param merged_data: main dataframe used in the program
    """
    col_1 = merged_data.pop('ROY')
    merged_data.insert(35, 'ROY', col_1)
    col_2 = merged_data.pop('MVPs')
    merged_data.insert(36, 'MVPs', col_2)
    col_3 = merged_data.pop('PS_MVPs')
    merged_data.insert(37, 'PS_MVPs', col_3)


def award_projections(merged_data, hof_age_mean, act_temp):
    """

    Using subjective multiplier and probability values, the number of some specific
    awards is calculated for a projection of what current active players could retire with.

    :param merged_data: main dataframe used in the program
    :param hof_age_mean: baseline value of when a player "retires" (at least statistically)
    :param act_temp: active player dataframe
    """
    award_age = hof_age_mean - 1

    early_multiplier = 1.2
    prime_multiplier = 1.1
    late_multiplier = 0.8

    award_multipliers = {
        'GGs': 0.8,
        'SSs': 1.0,
        'Hank_Aaron': 0.8,
        'Bat_Titles': 0.6,
        'All_Stars': 1.1
    }

    for i in merged_data.index:
        if merged_data.loc[i, 'status'] == 'active':
            player = merged_data.loc[i, 'Player']
            seasons = act_temp[act_temp['Player'] == player]
            num_seasons = len(seasons)

            if num_seasons < 5:
                multiplier = early_multiplier
            elif num_seasons < 8:
                multiplier = prime_multiplier
            else:
                multiplier = late_multiplier

            for j in merged_data.columns[38:]:
                col_multiplier = award_multipliers.get(j, 1.0)
                curr_total = merged_data.loc[i, j]
                avg = int(curr_total) / num_seasons
                std_dev = curr_total / (num_seasons * (12 ** 0.5))
                remaining = award_age - (seasons['Age'].max())
                adjustment_factor = 0.25
                proj_num = (avg - (adjustment_factor * std_dev)) * remaining * multiplier * col_multiplier
                proj_num = curr_total + proj_num
                merged_data.loc[i, j] = round(proj_num)


def enough_seasons(players_active):
    """

    The active player dataframe is reduced in size so that only players with more than four
    seasons remain. This is done so that future projection of player statistics is calculated
    with a reasonable amount of data to pull from.

    :param players_active: active player dataframe
    :return: the filtered active player dataframe
    """
    players_active = players_active.fillna(0)

    conn = sqlite3.connect(':memory:')
    players_active.to_sql('players_active', conn, index=False, if_exists='replace')

    query = """
    SELECT *
    FROM players_active
    WHERE Player IN (
        SELECT Player
        FROM players_active
        GROUP BY Player
        HAVING COUNT(*) >= 4
    )
    """

    enough = pd.read_sql_query(query, conn)

    conn.close()
    enough = enough_seasons_helper(enough)
    return enough


def enough_seasons_helper(enough):
    """

    Helper function that re-formats the newly created dataframe.

    :param enough: filtered dataframe from enough_seasons function
    :return: the reformatted dataframe
    """
    enough = enough.drop(['WAR', 'Lg', 'Pos'], axis=1)
    enough = enough.rename(columns={'WAR.1': 'WAR'})
    new_cols = ['Player', 'Team', 'Age', 'G', 'PA', 'AB',
                'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA',
                'OBP', 'SLG', 'OPS', 'OPS+', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
                'WAR', 'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser', 'Rbaser + Rdp',
                'Rfield']
    enough = enough[new_cols]
    return enough


def create_projections(enough, hof_age_mean):
    """

    Using subjective multiplier and probability values, projected stats are calculated for
    each active player upon their eventual retirement. This will be used to help predict
    future hall-of-fame-level players upon the active player crop.

    :param enough: the filtered active player dataframe
    :param hof_age_mean: baseline value of when a player "retires" (at least statistically)
    :return: a list of lists containing each player and their projected stats
    """
    count = enough_seasons['Player'].unique()
    proj_data = []

    early_multiplier = 1.2
    prime_multiplier = 1.0
    late_multiplier = 0.8

    for player in count:
        player_pred = []
        player_pred.append(player)
        player_data = enough[enough['Player'] == player]
        sorted_df = player_data.sort_values(by='Age', ascending=True)
        sorted_df = sorted_df.drop(['BA', 'OBP', 'SLG', 'OPS', 'OPS+'], axis=1)
        num_seasons = len(player_data)

        if num_seasons < 5:
            multiplier = early_multiplier
        elif num_seasons < 8:
            multiplier = prime_multiplier
        else:
            multiplier = late_multiplier

        for j in sorted_df.columns[3:]:
            curr_total = sorted_df[j].sum()
            avg = sorted_df[j].rolling(window=4).mean().iloc[-1]
            std_dev = sorted_df[j].rolling(window=4).std().iloc[-1]
            remaining = hof_age_mean - (sorted_df['Age'].iloc[-1])
            adjustment_factor = 0.5
            proj_num = (avg - (adjustment_factor * std_dev)) * remaining * multiplier
            proj_sum = curr_total + proj_num
            player_pred.append(round(proj_sum))
        proj_data.append(player_pred)

    projection_df = format_projections(proj_data)
    return projection_df


def format_projections(proj_data):
    """

    Using the projected player data calculated in create_projections, a dataframe is created.
    Additionally, each player's slash-line is calculated for using the projection data.

    :param proj_data: a list of lists containing each player and their projected stats
    :return: the newly created player projection dataframe
    """
    final_cols = ['Player', 'G', 'PA', 'AB',
                  'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GIDP',
                  'HBP', 'SH', 'SF', 'IBB', 'WAR', 'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser',
                    'Rbaser + Rdp', 'Rbaser + Rdp', 'Rfield']

    projection_df = pd.DataFrame(proj_data, columns=final_cols)
    war = projection_df.pop('WAR')
    projection_df.insert(1, 'WAR', war)
    projection_df['BA'] = projection_df.apply(lambda row: calc_bat_avg(row['H'], row['AB']), axis=1)
    projection_df['OBP'] = projection_df.apply(
        lambda row: calc_obp(row['H'], row['BB'], row['HBP'], row['AB'], row['SF']), axis=1)
    projection_df['SLG'] = projection_df.apply(
        lambda row: calc_slug(row['1B'], row['2B'], row['3B'], row['HR'], row['AB']), axis=1)
    projection_df['OPS'] = projection_df.apply(lambda row: calc_ops(row['OBP'], row['SLG']), axis=1)
    return projection_df


def merge_curr_hof(players_hof, projection_df, retired_players):
    """

    Function formats the three inputted dataframes the same and merges them into one main
    dataframe for later use.

    :param players_hof: hall of fame player dataframe
    :param projection_df: projection player dataframe
    :param retired_players: retired player dataframe
    :return: main dataframe used in the program
    """
    first_column_1 = players_hof.iloc[:, 0]
    hof_2 = players_hof.drop(players_hof.columns[0], axis=1)
    int_cols_hof = hof_2.columns[:-4]
    hof_2[int_cols_hof] = hof_2[int_cols_hof].astype(int)
    float_cols_hof = hof_2.columns[-4:]
    hof_2[float_cols_hof] = hof_2[float_cols_hof].astype(float)
    hof_2.insert(0, 'Player', first_column_1)

    first_column_2 = projection_df.iloc[:, 0]
    temp_df = projection_df.drop(projection_df.columns[0], axis=1)
    int_cols_temp = temp_df.columns[:-4]
    temp_df[int_cols_temp] = temp_df[int_cols_temp].astype(int)
    float_cols_temp = temp_df.columns[-4:]
    temp_df[float_cols_temp] = temp_df[float_cols_temp].astype(float)
    temp_df.insert(0, 'Player', first_column_2)

    temp_data = pd.merge(hof_2, projection_df, how='outer')
    merged_data = merge_ret_helper(temp_data, retired_players)
    return merged_data


def merge_ret_helper(temp_data, retired_players):
    """

    Helper function that combines  the first merged dataframe from merge_curr_hof and the last
    desired dataframe to be merged.

    :param temp_data: first merged dataframe from merge_curr_hof
    :param retired_players: retired player dataframe
    :return: main dataframe used in the program
    """
    order_ret_cols = ['Player', 'WAR', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR',
                      'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
                      'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser', 'Rbaser + Rdp',
                      'Rfield', 'BA', 'OBP', 'SLG', 'OPS']
    retired_players = retired_players[order_ret_cols]

    first_column_3 = retired_players.iloc[:, 0]
    ret_2 = retired_players.drop(retired_players.columns[0], axis=1)
    ret_2 = ret_2.fillna(0)
    int_cols_ret = ret_2.columns[:-4]
    ret_2[int_cols_ret] = ret_2[int_cols_ret].astype(int)
    float_cols_ret = ret_2.columns[-4:]
    ret_2[float_cols_ret] = ret_2[float_cols_ret].astype(float)
    ret_2.insert(0, 'Player', first_column_3)
    ret_2 = ret_2[ret_2['G'] >= 1000]
    ret_2 = ret_2[ret_2['WAR'] >= 35]
    merged_data = pd.merge(temp_data, ret_2, how='outer')
    return merged_data


def determine_status(merged_data, players_hof, retired_players):
    """

    Transversing through the main dataframe of the program, the function determines each
    player's activity status, which will be used for predictions later on.

    :param merged_data: main dataframe used in the program
    :param players_hof: hall of fame player dataframe
    :param retired_players: retired player dataframe
    """
    merged_data["status"] = None

    hof_check = []
    for i in range(len(players_hof)):
        star = players_hof.at[i, 'Player']
        hof_check.append(star)

    retired_check = []
    for i in range(len(retired_players)):
        star = retired_players.at[i, 'Player']
        retired_check.append(star)

    for i in range(len(merged_data)):
        player = merged_data.at[i, 'Player']
        if player in hof_check:
            merged_data.at[i, 'status'] = 'hof'
        elif player in retired_check:
            merged_data.at[i, 'status'] = 'retired'
        else:
            merged_data.at[i, 'status'] = 'active'


def get_model_params(merged_data):
    """

    Function finds the best parameters for a Random Forest Classifier model.

    :param merged_data: main dataframe used in the program
    :return: the best parameters for the desired model, preprocessing pipeline
    """
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

    log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                 FunctionTransformer(log_transform, feature_names_out="one-to-one"), StandardScaler())

    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    preprocessing = ColumnTransformer([
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ("log", log_pipeline, make_column_selector(dtype_include=np.number)),

    ],
        remainder=default_num_pipeline)

    x_cols = ['WAR', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR',
              'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
              'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser', 'Rbaser + Rdp',
              'Rfield', 'BA', 'OBP', 'SLG', 'OPS', 'status', 'ROY', 'MVPs', 'PS_MVPs',
              'GGs', 'SSs', 'Hank_Aaron', 'Bat_Titles', 'All_Stars']

    y_col = 'status'

    x_data = merged_data[x_cols].copy()
    x_data = x_data.fillna(0)

    y_data = merged_data[y_col].copy()
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)

    curr_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("scaler", StandardScaler()),
        ("random_forest", RandomForestClassifier(random_state=42))
    ])

    random_search = RandomizedSearchCV(
        curr_pipeline,
        param_distributions={
            "random_forest__n_estimators": randint(100, 300),
            "random_forest__max_depth": [None, 10, 20, 30],
            "random_forest__min_samples_split": [2, 5, 10],
            "random_forest__min_samples_leaf": [1, 2, 4],
            "random_forest__max_features": ["sqrt", "log2", None]
        },
        n_iter=50,
        cv=10,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    random_search.fit(x_data, y_data)
    best_params = random_search.best_params_
    return best_params, preprocessing


def create_model(best_params, preprocessing, merged_data):
    """

    Using the parameters created in get_model_params, a Random Forest Classifier model
    is created and fitted to predict 'status' values for current active players in the
    database.

    :param best_params: the best parameters for the desired model
    :param preprocessing: preprocessing pipeline
    :param merged_data: main dataframe used in the program
    :return: fitted prediction model, copy of the main dataframe
    """
    best_rf = RandomForestClassifier(
        max_depth=best_params['random_forest__max_depth'],
        max_features=best_params['random_forest__max_features'],
        min_samples_leaf=best_params['random_forest__min_samples_leaf'],
        min_samples_split=best_params['random_forest__min_samples_split'],
        n_estimators=best_params['random_forest__n_estimators'],
        class_weight='balanced',
        random_state=42
    )

    final_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("scaler", StandardScaler()),
        ("random_forest", best_rf)
    ])

    x_cols = ['WAR', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR',
              'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
              'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser', 'Rbaser + Rdp',
              'Rfield', 'BA', 'OBP', 'SLG', 'OPS', 'status', 'ROY', 'MVPs', 'PS_MVPs',
              'GGs', 'SSs', 'Hank_Aaron', 'Bat_Titles', 'All_Stars']

    rf_data = merged_data.copy()
    label_encoder = LabelEncoder()
    rf_data['status_encoded'] = label_encoder.fit_transform(rf_data['status'])
    new_y_col = 'status_encoded'

    train = rf_data[rf_data['status'] != 'active']
    final_pipeline.fit(train[x_cols], train[new_y_col])
    return final_pipeline, rf_data


def get_predictions(final_pipeline, rf_data):
    """

    A prediction dataframe is created, using the fitted prediction model, of current active players
    who have the potential to reach hall of fame status, as their current career projects.

    :param final_pipeline: fitted prediction model
    :param rf_data: copy of the main dataframe
    :return: prediction dataframe
    """
    x_cols = ['WAR', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR',
              'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
              'WAA', 'oWAR', 'dWAR', 'Rbat', 'Rdp', 'Rbaser', 'Rbaser + Rdp',
              'Rfield', 'BA', 'OBP', 'SLG', 'OPS', 'status', 'ROY', 'MVPs', 'PS_MVPs',
              'GGs', 'SSs', 'Hank_Aaron', 'Bat_Titles', 'All_Stars']

    new_y_col = 'status_encoded'

    train = rf_data[rf_data['status'] != 'active']
    test = rf_data[rf_data['status'] == 'active']
    final_pipeline.fit(train[x_cols], train[new_y_col])
    predictions = final_pipeline.predict(test[x_cols])
    predictions = predictions.round().astype(int)
    predictions_df = pd.Series(predictions, index=test.index, name='predictions')
    rf_data['predictions'] = predictions_df
    hof_predictions = rf_data[(rf_data['predictions'] == 1) & (test['WAR'] >= 56)]
    hof_predictions.drop('status_encoded', axis=1)
    return hof_predictions


def main():
    players_hof = read_and_encode_1('Hall of Fame.csv')
    retired_players = read_and_encode_1('Retired.csv')
    players_active = read_and_encode_1('Active Player Seasons.csv')
    all_stars = read_and_encode_1('All Stars.csv')

    al_gg = read_and_encode_2('AL Gold Glove Winners.csv')
    al_ss = read_and_encode_2('AL Silver Slugger Winners.csv')
    nl_gg = read_and_encode_2('NL Gold Glove Winners.csv')
    nl_ss = read_and_encode_2('NL Silver Slugger Winners.csv')
    bat_titles = read_and_encode_2('Batting Title Winners.csv')
    roy_winners = read_and_encode_2('ROY Winners.csv')
    hank_aarons = read_and_encode_2('Hank Aaron Award Winners.csv')
    mvp_winners = read_and_encode_2('MVP Winners.csv')
    ps_mvps = read_and_encode_2('Post Season MVP Winners.csv')

    enough = enough_seasons(players_active)
    players_hof, hof_age_mean = fix_age(players_hof)
    projection_df = create_projections(enough, hof_age_mean)
    merged_data = merge_curr_hof(players_hof, projection_df, retired_players)
    determine_status(merged_data, players_hof, retired_players)

    hof_temp = read_and_encode_1('Hall of Fame.csv', check=True)
    act_temp = read_and_encode_1('Active Player Seasons.csv', check=True)
    ret_temp = read_and_encode_1('Retired.csv', check=True)

    setup_award(merged_data, 'GGs', al_gg, nl_gg, hof_temp, act_temp, ret_temp)
    setup_award(merged_data, 'SSs', al_ss, nl_ss, hof_temp, act_temp, ret_temp)
    simple_awards(merged_data, roy_winners, 'ROY', count=False)
    simple_awards(merged_data, hank_aarons, 'Hank_Aaron', count=True)
    simple_awards(merged_data, mvp_winners, 'MVPs', count=True)
    ps_awards(merged_data, ps_mvps)
    bt_create(merged_data, bat_titles)
    as_create(merged_data, all_stars)
    format_after_awards(merged_data)
    award_projections(merged_data, hof_age_mean, act_temp)

    best_params, preprocessing = get_model_params(merged_data)
    final_pipeline, rf_data = create_model(best_params, preprocessing, merged_data)
    hof_predictions = get_predictions(final_pipeline, rf_data)
    print(hof_predictions)


main()
