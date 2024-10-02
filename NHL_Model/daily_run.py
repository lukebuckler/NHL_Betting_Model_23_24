#access the website to scrape the current season's games and data
import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
from datetime import datetime
import os
import re
import datetime
#Scrape the betting odds for each team for the day
url = 'https://sports.yahoo.com/nhl/odds/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

tables = soup.find_all('table')
data = []

for table in tables:
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)

df = pd.DataFrame(data)
filtered_df = df[df[0].astype(str).map(len) > 5]

filtered_df['Team'] = filtered_df[0].apply(lambda x: x.split('(')[0].strip())

filtered_df['Odds'] = filtered_df[1].apply(lambda x: x[-4:].strip())

new_df = filtered_df[['Team', 'Odds']]

# use the mapping dictionary to replace the team names with the abbreviations
abbreviations_df = pd.read_csv('teams.csv')
mapping_dict = dict(zip(abbreviations_df['Team'], abbreviations_df['Abbrv']))
new_df['Team'] = new_df['Team'].map(mapping_dict)

#convert the american odds into decimal odds, and then into implied probabilities
def convert_american_to_decimal(american_odds):
    if american_odds.startswith('+'):
        return 1 + int(american_odds[1:]) / 100
    elif american_odds.startswith('-'):
        return 1 + 100 / abs(int(american_odds))
    else:
        return None


new_df['Decimal Odds'] = new_df['Odds'].apply(convert_american_to_decimal)
new_df['Decimal Odds'] = new_df['Decimal Odds'].round(2)

def decimal_to_implied_probability(decimal_odds):
    if decimal_odds > 0:
        return round(1 / decimal_odds, 2)
    else:
        return None 

new_df['Implied Probability'] = new_df['Decimal Odds'].apply(decimal_to_implied_probability)
new_df.to_csv('nhl_odds.csv', index=False)

#wesbite with current season's games and stats
url = 'https://www.naturalstattrick.com/games.php?fromseason=20232024&thruseason=20232024&stype=2&sit=all&loc=B&team=All&rate=n'

response = requests.get(url)

#use beautiful soup to parse the webpage's content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table
table = soup.find('table')

# Read the table using pandas
df = pd.read_html(str(table))[0]
#drop all columns with a %
cols_to_drop = [col for col in df.columns if '%' in col]
    
#drop other unneccesary columns
df.drop(columns=cols_to_drop, inplace=True)
df.drop('Unnamed: 2', axis=1, inplace=True)
df.drop('Attendance', axis=1, inplace=True)
df.drop('TOI', axis=1, inplace=True)
df.to_csv('23_24_Games/23_24_games.csv', index=False)

# use the mapping dictionary to replace the team names with the abbreviations
abbreviations_df = pd.read_csv('teams.csv')
mapping_dict = dict(zip(abbreviations_df['Team'], abbreviations_df['Abbrv']))

def update_and_overwrite_file(file_path, mapping_dict):
    df = pd.read_csv(file_path)
    df['Team'] = df['Team'].map(mapping_dict)
    df.to_csv(file_path, index=False)  

file_paths = ['23_24_Games/23_24_games.csv']

for file_path in file_paths:
    update_and_overwrite_file(file_path, mapping_dict)

# use the mapping dictionary to replace the team names in the game string with the abbreviations
abbreviations_df = pd.read_csv('teams.csv')
mapping_dict = dict(zip(abbreviations_df['Name'], abbreviations_df['Abbrv']))

def replace_team_names(game_string, mapping_dict):
    pattern = r'(\d{4}-\d{2}-\d{2}) - ([\w\s]+) (\d+), ([\w\s]+) (\d+)'
    match = re.match(pattern, game_string)

    if match:
        date, team1, score1, team2, score2 = match.groups()
        team1_abbr = mapping_dict.get(team1.strip(), team1)
        team2_abbr = mapping_dict.get(team2.strip(), team2) 
        return f"{date} - {team1_abbr} {score1}, {team2_abbr} {score2}"
    else:
        return game_string

def update_and_overwrite_file(file_path, mapping_dict):
    df = pd.read_csv(file_path)
    df['Game'] = df['Game'].apply(lambda x: replace_team_names(x, mapping_dict))
    df.to_csv(file_path, index=False)

file_paths = ['23_24_Games/23_24_games.csv']

for file_path in file_paths:
    update_and_overwrite_file(file_path, mapping_dict)

input_directory = "C:/Users/luken/Desktop/NHL_Model/23_24_Games/"


base_output_directory = "C:/Users/luken/Desktop/NHL_Model/23_24_Team"
#iterate through the files and create new files for each team's games for each year
for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):
        year = file_name.split('_games.csv')[0]

        year_directory = base_output_directory
        if not os.path.exists(year_directory):
            os.makedirs(year_directory)

        file_path = os.path.join(input_directory, file_name)
        df = pd.read_csv(file_path)
        pattern = r'(\d{4}-\d{2}-\d{2}) - ([\w\s]+) (\d+), ([\w\s]+) (\d+)'
        df[['Date', 'AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore']] = df['Game'].str.extract(pattern)
        df['HomeResult'] = 'Draw'
        df.loc[df['HomeScore'] > df['AwayScore'], 'HomeResult'] = 'Won'
        df.loc[df['HomeScore'] < df['AwayScore'], 'HomeResult'] = 'Lost'
        

#sort each team's games into a it's own file for the current season
        team_subsets = {team: df[df['Team'] == team] for team in df['Team'].unique()}

        for team, subset in team_subsets.items():
            team_file_name = f"{team}.csv"
            team_file_path = os.path.join(year_directory, team_file_name)
            subset.to_csv(team_file_path, index=False)

#calculate days of rest
dir_path = "C:/Users/luken/Desktop/NHL_Model/23_24_Team"


all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]

for file in all_files:
    file_path = os.path.join(dir_path, file)
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        date_col = 'Date'
    df[date_col] = pd.to_datetime(df[date_col])

    df['time_diff'] = df[date_col].diff()
    df['days_of_rest'] = df['time_diff'].dt.days - 1
    df.drop('time_diff', axis=1, inplace=True)
    df.to_csv(file_path, index=False)

#from the schedule in the file '23_24_schedule' create a dataframe that has all of the games for the current day
current_date = datetime.date.today()
print(current_date)
df = pd.read_csv('23_24_sched.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.date
df2 = df[df['Date'] == current_date]
td = df2[['Game','AwayTeam','HomeTeam']]
td.to_csv("today.csv", index=False)

#do the rolling average average but in order to get the data for the last 10 games, we perform it wihtout  shifting by one to include the current games
#then take the last row from each team's files and create a new file that has the most recent stats for every team so pull from if they play today
current_date = pd.Timestamp(datetime.date.today())

def calculate_rolling_average(df, columns, window=10):
    rolling_df = df[columns].rolling(window=window, min_periods=10).mean()
    return rolling_df

#Base path for the original data 
base_path = '23_24_Team'
df4 = pd.DataFrame()
# New top-level folder for averaged data
average_data_path = '23_24_test'
os.makedirs(average_data_path, exist_ok=True)
# Iterate over each year's folder
# Iterate over each team's CSV file
for team_file in os.listdir(base_path):
    if team_file.endswith('.csv'):
        team_path = os.path.join(base_path, team_file)
        # Read the CSV file
        df = pd.read_csv(team_path)

        # Select the desired columns and calculate the rolling averages
        selected_columns = ['Game', 'Team', 'Date', 'AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore', 'HomeResult', 'days_of_rest']
        other_columns = df.columns.difference(selected_columns)
        rolling_df = calculate_rolling_average(df, other_columns)

    
        # Combine the selected columns with the rolling averages
        combined_df = pd.concat([df[selected_columns], rolling_df], axis=1)
        cols_to_drop = ['Game','AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore', 'HomeResult','days_of_rest']
       
    #drop other unneccesary columns
        combined_df.drop(columns=cols_to_drop, inplace=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df['time_diff']=(current_date-combined_df['Date'])
        combined_df['days_of_rest'] = combined_df['time_diff'].dt.days - 1
        combined_df.drop('time_diff', axis=1, inplace=True)
        combined_df.insert(1, 'days_of_rest', combined_df.pop('days_of_rest'))
        last_row_df = combined_df.tail(1)
        df4 = pd.concat([df4, last_row_df], ignore_index=True, axis=0)
        # Save the new DataFrame as a CSV file in the year-specific folder under 'average_data'

df4.to_csv('test.csv', index=False)

#pull form the file we just created and the list of today's game and get the stats for the home and away teams with the prefixes added and save it to the "todays_games.csv"
df_test = pd.read_csv('test.csv')
df_today= pd.read_csv('today.csv')

def add_prefix_to_columns(df, prefix):
    df.columns = [prefix + col for col in df.columns]
    return df

def find_team_data_modified(team, df_test):
    team_data = df_test[df_test['Team'] == team].drop(columns=['Team', 'Date'])
    return team_data
combined_data = []
for index, row in df_today.iterrows():
    game_info = row['Game']
    away_team = row['AwayTeam']
    home_team = row['HomeTeam']
    away_team_data = add_prefix_to_columns(find_team_data_modified(away_team, df_test).copy(), 'a_')
    home_team_data = add_prefix_to_columns(find_team_data_modified(home_team, df_test).copy(), 'h_')
    if not away_team_data.empty:
        away_team_data = away_team_data.iloc[0]
    if not home_team_data.empty:
        home_team_data = home_team_data.iloc[0]

    combined_row = pd.concat([away_team_data, home_team_data])
    combined_row['Game'] = game_info
    combined_data.append(combined_row)
df_combined_corrected = pd.DataFrame(combined_data)
df_combined_corrected.to_csv('today_games.csv', index=False)

svm_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
df5 = pd.read_csv('today_games.csv')
games = df5['Game']
features= df5.drop(columns=['Game'], inplace=True)


#Load the model and scaler
svm_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
df5 = pd.read_csv('today_games.csv')

game_ids = df5['Game']

features = df5.drop(columns=['Game'])

#Scale features
scaled_features = scaler.transform(features)

#Predict probabilities
probabilities = svm_model.predict_proba(scaled_features)
predicted_labels = svm_model.predict(scaled_features)
#Create a dataframe for the output
prob_df = pd.DataFrame(probabilities, columns=[f'Prob_{label}' for label in svm_model.classes_])
prob_df['Predicted Label'] = predicted_labels
prob_df['Game ID'] = game_ids

# Assuming df is your DataFrame with games data
# Splitting the 'Game ID' column to extract away and home team names
prob_df[['Date', 'Teams']] = prob_df['Game ID'].str.split(' - ', expand=True)
prob_df[['Away Team', 'Home Team']] = prob_df['Teams'].str.split(', ', expand=True)
prob_df.drop(columns=['Game ID', 'Teams'], inplace=True)

# Load the NHL odds data
nhl_odds_df = pd.read_csv('nhl_odds.csv')

# Merging the odds information for the away team
prob_df = pd.merge(prob_df, nhl_odds_df, left_on='Away Team', right_on='Team', how='left')
prob_df.rename(columns={'Odds': 'Away Team Odds', 'Decimal Odds': 'Away Team Decimal Odds', 
                   'Implied Probability': 'Away Team Implied Probability'}, inplace=True)

# Merging the odds information for the home team
prob_df = pd.merge(prob_df, nhl_odds_df, left_on='Home Team', right_on='Team', how='left')
prob_df.rename(columns={'Odds': 'Home Team Odds', 'Decimal Odds': 'Home Team Decimal Odds', 
                   'Implied Probability': 'Home Team Implied Probability'}, inplace=True)
prob_df.drop(columns=['Team_x', 'Team_y'], inplace=True)

prob_df = prob_df[['Date', 'Away Team', 'Away Team Odds', 'Away Team Decimal Odds', 'Away Team Implied Probability', 
         'Prob_0', 'Prob_1', 'Home Team Implied Probability', 'Home Team Decimal Odds', 'Home Team Odds', 'Home Team']]
prob_df.rename(columns={'Prob_0': 'Away Probability', 'Prob_1': 'Home Probability'}, inplace=True)




#save it to a csv
#prob_df.to_csv('predicted_probabilities_and_labels.csv', index=False)
formatted_date = current_date.strftime('%Y-%m-%d')
folder_name = "23_24_results"
filename = f"{folder_name}/{formatted_date}.csv"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


prob_df.to_csv(filename, index=False)
print("Results can be found in 23_24_results in the file called " + filename)