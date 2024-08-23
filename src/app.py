import nfl_data_py as nfl
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_daq as daq
import webbrowser
from flask import Flask
import json
from threading import Timer
import plotly.graph_objects as go
import plotly.express as px


ids = nfl.import_ids()
ids = ids[~ids['gsis_id'].isna()]
ids = ids[ids['position'].isin(['QB', 'RB', 'WR', 'TE'])]
ids = ids[['gsis_id', 'name', 'college']]
ids.rename(columns={'gsis_id': 'id'}, inplace=True)
ids.set_index('id', drop=True, inplace=True)

kYears = np.arange(2012, 2024, 1).tolist()
weekly = nfl.import_weekly_data(kYears)
weekly = weekly[weekly['season_type'] == 'REG']
weekly.set_index('player_id', drop=True, inplace=True)

weekly_df = pd.merge(ids, weekly, left_index=True, right_index=True)
weekly_df.reset_index(inplace=True)
weekly_df.fillna(0, inplace=True)
weekly_df.rename(columns={'player_id': 'id', 'fantasy_points_ppr': 'ppr_points'}, inplace=True)

excel_file = 'FF_Predictions_Aug23.xlsx'
qb_df = pd.read_excel(excel_file, sheet_name='QB')
rb_df = pd.read_excel(excel_file, sheet_name='RB')
wr_df = pd.read_excel(excel_file, sheet_name='WR')
te_df = pd.read_excel(excel_file, sheet_name='TE')

def get_season_data(pos):
    season = pd.read_excel('FF_Season_Data_Aug16.xlsx', sheet_name=pos)
    season.sort_values(by=['name'], inplace=True)
    season['id'] = season.groupby('name')['id'].transform(lambda x: x.ffill())
    season['id'] = season.groupby('name')['id'].transform(lambda x: x.bfill())
    return season

qb_season = get_season_data('QB')
rb_season = get_season_data('RB')
wr_season = get_season_data('WR')
te_season = get_season_data('TE')

tmDesc = nfl.import_team_desc()

for i, df in enumerate([qb_df, rb_df, wr_df, te_df]):
    original_index = df.index  # Save the original index
    merged_df = pd.merge(df, tmDesc[['team_abbr', 'team_name', 'team_logo_espn', 'team_conf', 'team_division', 'team_color', 'team_color2', 'team_color3', 'team_color4']],
                         left_on='team', right_on='team_abbr', how='left')
    merged_df.index = original_index  # Restore the original index
    # Reassign the merged DataFrame back to the respective variables
    if i == 0:
        qb_df = merged_df
    elif i == 1:
        rb_df = merged_df
    elif i == 2:
        wr_df = merged_df
    elif i == 3:
        te_df = merged_df

all_pos_df = pd.concat([qb_df, rb_df, wr_df, te_df], ignore_index=True)
all_season = pd.concat([qb_season, rb_season, wr_season, te_season], ignore_index=True)

all_pos_df['position'] = all_pos_df['ADP_rank'].apply(lambda x: x[:2])  # Gets pos (QB, RB, WR, TE)

season_data = {
    'QB': qb_season,
    'RB': rb_season,
    'WR': wr_season,
    'TE': te_season
}

def extract_numeric_rank(rank):
    return int(''.join(filter(str.isdigit, rank)))

all_pos_df['ADP_Rank_Num'] = all_pos_df['ADP_rank'].apply(extract_numeric_rank)
all_pos_df['Predicted_Rank_Num'] = all_pos_df['predicted_rank'].apply(extract_numeric_rank)

def get_team_stats_from_non_rookie(team_abbr):
    # Check for non-rookie players in the same position
    for df in [qb_df, rb_df, wr_df, te_df]:
        non_rookie = df[(df['team'] == team_abbr) & (df['years_exp'] > 0)]
        if not non_rookie.empty:
            # Return the stats from the first non-rookie found
            return non_rookie.iloc[0][['prev_wins', 'prev_team_rank', 'prev_pass_rank', 'prev_rush_rank']]
    
    # If no non-rookies are found, return default values (or handle as needed)
    return pd.Series({'prev_wins': 'N/A', 'prev_team_rank': 'N/A', 'prev_pass_rank': 'N/A', 'prev_rush_rank': 'N/A'})

def determine_experience_color(position, years_exp):
    if position == 'QB':
        if years_exp == 0:
            return 'red'
        elif years_exp >= 10:
            return 'yellow'
        else:
            return 'green'
    elif position == 'RB':
        if years_exp >= 9:
            return 'red'
        elif 6 <= years_exp <= 8:
            return 'yellow'
        else:
            return 'green'
    elif position == 'WR':
        if years_exp >= 12:
            return 'red'
        elif 8 <= years_exp <= 11 or years_exp == 0:
            return 'yellow'
        else:
            return 'green'
    elif position == 'TE':
        if years_exp == 0:
            return 'red'
        elif years_exp >= 12:
            return 'yellow'
        else:
            return 'green'
    return '#d3d3d3' 

def determine_best_season_color(position, best_rank):
    if isinstance(best_rank, str):
        return '#d3d3d3'  # Neutral gray for N/A or similar non-numeric values
    if position == 'QB':
        if best_rank >= 25:
            return 'red'
        elif 9 <= best_rank <= 24:
            return 'yellow'
        else:
            return 'green'
    elif position == 'RB':
        if best_rank >= 36:
            return 'red'
        elif 13 <= best_rank <= 35:
            return 'yellow'
        else:
            return 'green'
    elif position == 'WR':
        if best_rank >= 42:
            return 'red'
        elif 16 <= best_rank <= 41:
            return 'yellow'
        else:
            return 'green'
    elif position == 'TE':
        if best_rank >= 20:
            return 'red'
        elif 9 <= best_rank <= 19:
            return 'yellow'
        else:
            return 'green'
    return '#d3d3d3'  # Default to neutral gray

def determine_avg_weekly_points_color(position, weekly_ppr_points):
    if isinstance(weekly_ppr_points, str):
        return '#d3d3d3'  # Neutral gray for N/A or similar non-numeric values
    if position == 'QB':
        if weekly_ppr_points <= 15:
            return 'red'
        elif 15 < weekly_ppr_points <= 21:
            return 'yellow'
        else:
            return 'green'
    elif position == 'RB':
        if weekly_ppr_points <= 7:
            return 'red'
        elif 8 <= weekly_ppr_points <= 15:
            return 'yellow'
        else:
            return 'green'
    elif position == 'WR':
        if weekly_ppr_points <= 8:
            return 'red'
        elif 9 <= weekly_ppr_points <= 15:
            return 'yellow'
        else:
            return 'green'
    elif position == 'TE':
        if weekly_ppr_points <= 6:
            return 'red'
        elif 7 <= weekly_ppr_points <= 12:
            return 'yellow'
        else:
            return 'green'
    return '#d3d3d3'  # Default to neutral gray

def determine_competition_color(position, depth, team_adps):
    if position == 'QB':
        if depth == 'QB2':
            return 'red'
        elif depth == 'QB1' and team_adps['QB2'] <= team_adps['QB1'] + 75:
            return 'yellow'
        else:
            return 'green'
    elif position == 'RB':
        if depth == 'RB2' and team_adps['RB1'] < team_adps['RB2'] - 75:
            return 'red'
        elif depth in ['RB2', 'RB1'] and (team_adps['RB1'] <= team_adps[depth] - 75) or (team_adps['RB2'] <= team_adps[depth] + 75):
            return 'yellow'
        else:
            return 'green'
    elif position in ['WR', 'TE']:
        if (depth in ['WR3', 'WR2'] and (team_adps['TE1'] < team_adps[depth] or team_adps['WR2'] <= team_adps[depth] - 50)) or (depth == 'TE1' and team_adps['WR2'] < team_adps['TE1']):
            return 'red'
        elif (depth == 'WR3') or (depth == 'WR2' and team_adps['WR1'] <= team_adps[depth] - 75) or (depth == 'TE1' and team_adps['WR1'] <= team_adps['TE1'] - 75):
            return 'yellow'
        elif depth == 'WR1' and (team_adps['WR2'] <= team_adps['WR1'] + 50 and (team_adps['TE1'] <= team_adps['WR1'] + 75 or team_adps['WR3'] <= team_adps['WR1'] + 75)):
            return 'yellow'
        else:
            return 'green'

    return '#d3d3d3'  # Default to neutral gray

def determine_color(value, thresholds, reverse=False):
    # Handle non-numeric values like 'N/A'
    if isinstance(value, str) or pd.isna(value):
        return '#d3d3d3'  # Neutral gray for N/A values

    if reverse:
        red, yellow, green = thresholds
        if value <= green:
            return 'green'
        elif value <= yellow:
            return 'yellow'
        else:
            return 'red'
    else:
        red, yellow, green = thresholds
        if value <= red:
            return 'red'
        elif value <= yellow:
            return 'yellow'
        else:
            return 'green'

def calculate_gauges(position, player_info, best_rank, best_weekly_ppr_points, consistency_grade, team_adps, availability, explosiveness=None, production=None):
    color_scores_upside = {'green': .94, 'yellow': 0.6, 'red': 0.25, '#d3d3d3': 0.3}
    color_scores_risk = {'green': 0.15, 'yellow': 0.65, 'red': 0.85, '#d3d3d3': 0.25}

    is_rookie = player_info['years_exp'] == 0
    
    if is_rookie:
        if position == 'QB':
            rookie_bonus = .3
            if player_info['Predicted_Rank_Num'] <= 10:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 16:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 24:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
            
            if player_info['ADP_Rank_Num'] <= 8:
                adp_rank_score = 1
            elif player_info['ADP_Rank_Num'] <= 14:
                adp_rank_score = 0.75
            elif player_info['ADP_Rank_Num'] <= 20:
                adp_rank_score = 0.5
            else:
                adp_rank_score = 0.25    
            
        elif position == 'RB':
            rookie_bonus = .85
            if player_info['Predicted_Rank_Num'] <= 16:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 32:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 48:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
                
            if player_info['ADP_Rank_Num'] <= 12:
                adp_rank_score = 1
            elif player_info['ADP_Rank_Num'] <= 24:
                adp_rank_score = 0.75
            elif player_info['ADP_Rank_Num'] <= 36:
                adp_rank_score = 0.5
            else:
                adp_rank_score = 0.25
                
        elif position == 'WR':
            rookie_bonus = .75
            if player_info['Predicted_Rank_Num'] <= 24:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 48:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 72:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
                
            if player_info['ADP_Rank_Num'] <= 12:
                adp_rank_score = 1
            elif player_info['ADP_Rank_Num'] <= 24:
                adp_rank_score = 0.75
            elif player_info['ADP_Rank_Num'] <= 36:
                adp_rank_score = 0.5
            else:
                adp_rank_score = 0.25
        elif position == 'TE':
            rookie_bonus = .5
            if player_info['Predicted_Rank_Num'] <= 8:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 16:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 24:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
                
            if player_info['ADP_Rank_Num'] <= 6:
                adp_rank_score = 1
            elif player_info['ADP_Rank_Num'] <= 12:
                adp_rank_score = 0.75
            elif player_info['ADP_Rank_Num'] <= 16:
                adp_rank_score = 0.5
            else:
                adp_rank_score = 0.25
        
        upside = (0.2 * projected_rank_score + 0.55 * adp_rank_score + .25 * rookie_bonus) * 10 # Use a simple formula for rookies

        team_proj_score = color_scores_risk[determine_color(player_info['projected_wins'], (5, 9, 10))]
        competition_score = color_scores_risk[determine_competition_color(position, player_info['Depth'], team_adps)]
        oline_score = color_scores_risk[determine_color(player_info['line_rating'], (22, 21, 10), True)]
        risk = (0.4 * team_proj_score + 0.4 * competition_score + 0.2 * oline_score) * 10
    else:
        # Determine scores for each component used in the Upside gauge
        best_weekly_points_score = color_scores_upside[determine_avg_weekly_points_color(position, best_weekly_ppr_points)]
        explosiveness_score = color_scores_upside[determine_color(explosiveness, (10, 30, 50))]
        production_score = color_scores_upside[calculate_and_color_production(player_info['id'], position)]
        competition_score = color_scores_upside[determine_competition_color(position, player_info['Depth'], team_adps)]
        team_proj_score = color_scores_upside[determine_color(player_info['projected_wins'], (5, 9, 10))]
        
        if position == 'QB':
            if player_info['Predicted_Rank_Num'] <= 5:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 8:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 16:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
        elif position == 'RB':
            if player_info['Predicted_Rank_Num'] <= 5:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 12:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 24:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
        elif position == 'WR':
            if player_info['Predicted_Rank_Num'] <= 5:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 16:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 32:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
        elif position == 'TE':
            if player_info['Predicted_Rank_Num'] <= 5:
                projected_rank_score = 1
            elif player_info['Predicted_Rank_Num'] <= 8:
                projected_rank_score = 0.75
            elif player_info['Predicted_Rank_Num'] <= 16:
                projected_rank_score = 0.5
            else:
                projected_rank_score = 0.25
        
        # Calculate the Upside score using the updated weights
        upside = (
            0.15 * team_proj_score +
            0.15 * competition_score +
            0.10 * best_weekly_points_score +
            0.25 * explosiveness_score +
            0.25 * production_score +
            0.10 * projected_rank_score
        ) * 10  # Scale to a 0-10 range

        # Determine scores for each component used in the Risk gauge
        durability_score = color_scores_risk[determine_color(availability, (8, 14, 15))]
        oline_score = color_scores_risk[determine_color(player_info['line_rating'], (22, 21, 10), True)]
        team_change_score = color_scores_risk[determine_color(player_info['team_change'], (1, 1, 0), True)]
        consistency_grade_score = color_scores_risk[determine_color(consistency_grade, (.29, .69, .7))]
        experience_score = color_scores_risk[determine_experience_color(position, player_info['years_exp'])]
        competition_score = color_scores_risk[determine_competition_color(position, player_info['Depth'], team_adps)]
        
        # Calculate the Risk score using the updated weights
        risk = (
            0.10 * experience_score +
            0.15 * competition_score +
            0.40 * durability_score +
            0.15 * oline_score +
            0.05 * team_change_score +
            0.15 * consistency_grade_score
        ) * 10  # Scale to a 0-10 range
    
    return round(upside, 2), round(risk, 2)

def calculate_consistency(player_id, position):
    position_thresholds = {
        'QB': 16,
        'RB': 12,
        'WR': 12,
        'TE': 9
    }
    threshold = position_thresholds.get(position, 0)
    player_weekly = weekly_df[weekly_df['id'] == player_id]
    total_games = len(player_weekly)
    consistent_games = len(player_weekly[player_weekly['ppr_points'] >= threshold])
    return round((consistent_games / total_games), 2) if total_games > 0 else 0

def calculate_explosiveness(player_id, position):
    if position == 'QB':
        threshold = 25  # Example threshold for QB
        range = (10, 25, 40)
    elif position == 'RB':
        threshold = 20  # Example threshold for RB
        range = (10, 25, 50)
    elif position == 'WR':
        threshold = 20  # Example threshold for WR
        range = (10, 25, 50)
    elif position == 'TE':
        threshold = 15  # Example threshold for TE
        range = (10, 25, 50)

    # Filter the weekly_df by player_id
    player_weekly = weekly_df[weekly_df['id'] == player_id]
    total_games = len(player_weekly)
    explosive_games = len(player_weekly[player_weekly['ppr_points'] > threshold])

    if total_games == 0:
        return 0, range  # If the player has no games, return 0 for Explosiveness
    else:
        return round((explosive_games / total_games) * 100, 2), range  # Return as a percentage

def calculate_and_color_production(player_id, position):
    if position == 'QB':
        season_df = qb_season
        thresholds = (0.1, 0.3, 0.4)
    elif position == 'RB':
        season_df = rb_season
        thresholds = (0.1, 0.3, 0.4)
    elif position == 'WR':
        season_df = wr_season
        thresholds = (0.15, 0.25, 0.35)
    elif position == 'TE':
        season_df = te_season
        thresholds = (0.1, 0.2, 0.3)
    player_seasons = season_df[season_df['id'] == player_id]
    production = (player_seasons['ppr_points'].sum() / player_seasons['offense_snaps'].sum()) if player_seasons['offense_snaps'].sum() > 0 else 0
    return determine_color(production, thresholds)

def calculate_upside_risk_for_all_players():
    all_pos_df['upside'] = None
    all_pos_df['risk'] = None

    for index, player_info in all_pos_df.iterrows():
        position = player_info['position']
        player_id = player_info['id']

        # Use existing functions to calculate necessary stats
        best_rank = all_season[all_season['id'] == player_id]['player_rank'].min()
        best_weekly_ppr_points = all_season[all_season['id'] == player_id]['weekly_ppr_points'].max()
        consistency_grade = calculate_consistency(player_id, position)
        explosiveness, _ = calculate_explosiveness(player_id, position)
        production = calculate_and_color_production(player_id, position)
        availability = all_season[all_season['id'] == player_id]['games'].mean()
        team_abbr = player_info['team']

        team_adps = {
        'QB1': qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB1')]['ADP'].values[0] if len(qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB1')]) > 0 else 999,
        'QB2': qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB2')]['ADP'].values[0] if len(qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB2')]) > 0 else 999,
        'RB1': rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB1')]['ADP'].values[0] if len(rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB1')]) > 0 else 999,
        'RB2': rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB2')]['ADP'].values[0] if len(rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB2')]) > 0 else 999,
        'WR1': wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR1')]['ADP'].values[0] if len(wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR1')]) > 0 else 999,
        'WR2': wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR2')]['ADP'].values[0] if len(wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR2')]) > 0 else 999,
        'WR3': wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR3')]['ADP'].values[0] if len(wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR3')]) > 0 else 999,
        'TE1': te_df[(te_df['team'] == team_abbr) & (te_df['Depth'] == 'TE1')]['ADP'].values[0] if len(te_df[(te_df['team'] == team_abbr) & (te_df['Depth'] == 'TE1')]) > 0 else 999,
        }

        upside, risk = calculate_gauges(
            position, player_info, best_rank, best_weekly_ppr_points, consistency_grade, team_adps, availability, explosiveness, production
        )

        # Store the calculated values in the DataFrame
        all_pos_df.at[index, 'upside'] = upside
        all_pos_df.at[index, 'risk'] = risk

calculate_upside_risk_for_all_players()

def calculate_player_value_points(all_pos_df):
    df = pd.DataFrame(all_pos_df)

    # Normalize the player data for assigning value points
    max_predicted_points = df.groupby('position')['predicted_points'].transform('max')
    max_upside = df.groupby('position')['upside'].transform('max')
    min_risk = df.groupby('position')['risk'].transform('min')

    df['ADP_norm'] = 100 - (df['ADP'] / df['ADP'].max() * 100)
    df['predicted_points_norm'] = df['predicted_points'] / max_predicted_points * 100
    df['upside_norm'] = df['upside'] / max_upside * 100
    df['risk_norm'] = 100 - (df['risk'] / min_risk * 10)

    # Calculate value points based on the given criteria
    df['value_points'] = (
        df['ADP_norm'] * 0.75 +
        df['predicted_points_norm'] * 0.15 +
        df['upside_norm'] * 0.06 +
        df['risk_norm'] * 0.03
    )

    return df

all_pos_df = calculate_player_value_points(all_pos_df)

def subtract_slot(player_position, total_slots):
    if total_slots[player_position] > 0:
        total_slots[player_position] -= 1
        used_slot = player_position
    elif player_position in ['WR', 'RB'] and 'FLEX_WR' in total_slots and total_slots['FLEX_WR'] > 0:
        total_slots['FLEX_WR'] -= 1
        used_slot = 'FLEX_WR'
    elif player_position in ['WR', 'RB', 'TE'] and 'FLEX_WRT' in total_slots and total_slots['FLEX_WRT'] > 0:
        total_slots['FLEX_WRT'] -= 1
        used_slot = 'FLEX_WRT'
    else:
        total_slots['Bench'] -= 1
        used_slot = 'Bench'
    
    return used_slot, total_slots
    
def subtract_total_points(used_slot, player_value_points, total_value_points):
    # Subtract the player's value points from the correct position
    if used_slot in total_value_points:
        total_value_points[used_slot] -= player_value_points

    return total_value_points

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Store(id='is-typing', data=False),
    dcc.Store(id='total-value-points', data={}),
    dcc.Store(id='total-slots', data={}),
    dcc.Store(id='all_pos_df', data=all_pos_df.to_dict('records')),
    dcc.Store(id='player-recommendations', data=[]),
    dcc.Store(id='position-value-points', data={}),
    dcc.Store(id='user-drafted-players', data=[]),
    dcc.Store(id='drafted-players', data=[]),
    dcc.Store(id='player-value-points', data={}),
    dcc.Store(id='selected-player-index', data=0),
    dcc.Store(id='clicked-player-index', data=None),
    dcc.Store(id='clear-click-store', data=0),
    dcc.Store(id='selected-players', data=[]),
    dcc.Store(id='draft-board-data', data=[]),
    dcc.Store(id='draft-enabled', data=False),
    dcc.Store(id='previous-draft-board', data=[]),

    html.H1("Fantasy Football Draft Dashboard", style={'text-align': 'center', 'margin': '2vh 0'}),

    html.Div([
        html.Div([
            html.H3("Sort Players By"),
            html.Div([
                dcc.RadioItems(
                    id='sort-radio',
                    options=[
                        {'label': 'ADP', 'value': 'ADP'},
                        {'label': 'Draft Value', 'value': 'value_points'}
                    ],
                    value='ADP',
                    inline=True,
                    labelStyle={'margin-right': '10px'}
                ),
                dcc.Input(
                    id='search-bar',
                    type='text',
                    placeholder='Search by player name...',
                    style={'width': '55%', 'margin-left': '20px', 'pointer-events': 'none', 'background-color': '#f0f0f0'}
                ),
                html.Button("Hide Profile", id='toggle-profile', n_clicks=0, style={'margin-left': '20px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),
            
            html.Div([
                html.Span("Player", style={'width': '35%', 'font-weight': 'bold'}),
                html.Span("Team", style={'width': '15%', 'font-weight': 'bold'}),
                html.Span("ADP", style={'width': '15%', 'font-weight': 'bold'}),
                html.Span("Proj Rank", style={'width': '20%', 'font-weight': 'bold'}),
                html.Span("Draft", style={'width': '15%', 'font-weight': 'bold'}),
            ], style={'display': 'flex', 'padding': '2%', 'border-bottom': '0.3vh solid #ddd', 'width': '100%', 'box-sizing': 'border-box'}),
            
            html.Ul(id='player-list', style={
                'list-style-type': 'none', 
                'padding': '.5%', 
                'height': '80vh', 
                'overflow-y': 'scroll', 
                'border': '0.3vh solid #ddd',
                'margin': '0',
                'background-color': '#f9f9f9',
                'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                'border-radius': '0.5vh',
                'width': '100%',
                'box-sizing': 'border-box'
            })
        ], style={'width': '30%', 'height': '85vh', 'float': 'left', 'padding-right': '1%', 'box-sizing': 'border-box'}),
        
        html.Div([
            html.Div(id='profile-container', children=[
                html.H3("Player Profile", style={'text-align': 'center'}),
                html.Div([
                    html.Div(id='headshot-container', style={
                        'position': 'relative',
                        'width': '19.8%',  
                        'height': '20%',  
                        'border': '0.3vh solid #ddd',
                        'border-radius': '0.5vh',
                        'box-shadow': '0 0.2vh 0.4vh rgba(0, 0, 0, 0.1)',
                        'display': 'flex',
                        'align-items': 'center',
                        'justify-content': 'center',
                        'box-sizing': 'border-box',
                        'overflow': 'hidden',  
                        'background-color': '#fff'  
                    }),
                    html.Div(id='team-details', style={
                        'border': '0.3vh solid #ddd', 
                        'padding': '.5%',
                        'background-color': '#fff',
                        'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                        'border-radius': '0.5vh',
                        'box-sizing': 'border-box',
                        'width': '25%',  
                        'height': '25%',
                        'margin-left': '1%'
                    }),
                    html.Div(id='pros-cons-container', style={
                        'border': '0.3vh solid #ddd', 
                        'padding': '.5%',
                        'background-color': '#fff',
                        'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                        'border-radius': '0.5vh',
                        'box-sizing': 'border-box',
                        'width': '25%',  
                        'height': '25%',
                        'margin-left': '1%'  
                    }),
                    html.Div(id='projections-container', style={
                        'border': '0.3vh solid #ddd', 
                        'padding': '.5%',
                        'background-color': '#fff',
                        'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                        'border-radius': '0.5vh',
                        'box-sizing': 'border-box',
                        'width': '30%',  
                        'height': '25%',
                        'margin-left': '1%'  
                    }),
                ], style={'display': 'flex', 'flex-direction': 'row'}),  
                
                html.Div(id='player-details', style={
                    'border': '0.3vh solid #ddd', 
                    'padding': '.5%',
                    'background-color': '#fff',
                    'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                    'border-radius': '0.5vh',
                    'box-sizing': 'border-box',
                    'width': '19%',  
                    'height': '20.8vh',
                    'margin-top': '-21vh'  
                }),
                
                html.Div([
                    html.Div(id='previous-seasons', style={
                        'border': '0.3vh solid #ddd', 
                        'padding': '.5%',
                        'background-color': '#fff',
                        'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                        'border-radius': '0.5vh',
                        'box-sizing': 'border-box',
                        'width': '44%',  
                        'height': '60%',  
                        'float': 'left',
                        'margin-top': '1vh',
                        'min-height': '49vh',
                        'max-height': '49vh'
                    }, children=[
                        dcc.Tabs(id='season-tabs', value='General', children=[
                            dcc.Tab(label='General', value='General'),
                            dcc.Tab(label='Pass', value='Pass'),
                            dcc.Tab(label='Rush', value='Rush'),
                            dcc.Tab(label='Rec', value='Rec')
                        ]),
                        dcc.RangeSlider(id='season-slider', min=2000, max=2023, step=1, 
                                        marks={i: str(i) for i in range(2000, 2024)}, value=[2000, 2023]),
                        html.Div(id='season-stats-content', style={
                            'height': 'calc(100% - 120px)',  
                            'overflow-y': 'scroll',
                            'box-sizing': 'border-box'
                        })
                    ]),
                    html.Div([
                        html.Div([
                            html.Div([
                                dcc.Dropdown(
                                    id='season-filter',
                                    placeholder="Select a season",
                                    clearable=False,
                                    style={'width': '90%', 'display': 'inline-block', 'margin-right': '5%'}
                                ),
                                html.Button('×', id='clear-season-filter', n_clicks=0, style={
                                    'position': 'absolute',
                                    'right': '0',
                                    'top': '0',
                                    'width': '10%',
                                    'border': 'none',
                                    'background': 'none',
                                    'font-size': '20px',
                                    'color': 'black',
                                    'cursor': 'pointer',
                                    'padding': '0',
                                    'margin': '0'
                                })
                            ], style={'position': 'relative', 'width': '48%', 'display': 'inline-block'}),
                            
                            dcc.RadioItems(
                                id='points-filter',
                                options=[
                                    {'label': 'Total Fantasy Points', 'value': 'ppr_points'},
                                    {'label': 'Weekly Fantasy Points', 'value': 'weekly_ppr_points'}
                                ],
                                value='ppr_points',
                                inline=True,
                                labelStyle={'margin-right': '20px'},
                                style={'width': '48%', 'display': 'inline-block'}
                            )
                        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '1vh'}),
                        
                        dcc.Graph(id='season-points-graph')
                    ], style={
                        'border': '0.3vh solid #ddd', 
                        'padding': '.5%',
                        'background-color': '#fff',
                        'box-shadow': '0 0.4vh 0.8vh rgba(0, 0, 0, 0.1)',
                        'border-radius': '0.5vh',
                        'box-sizing': 'border-box',
                        'width': '55%', 
                        'float': 'right', 
                        'box-sizing': 'border-box', 
                        'height': '57%', 
                        'margin-top': '1vh',
                        'margin-left': '1%',
                        #'min-height': '49vh',
                        #'max-height': '49vh'
                    }) 
                ], style={'display': 'flex', 'width': '100%', 'box-sizing': 'border-box'})
            ], style={'display': 'block'}),  # Initially visible

            html.Div(id='draft-board-container', style={'display': 'none', 'width': '68%', 'float': 'left', 'box-sizing': 'border-box'}, children=[
                html.H3("Draft Setup", style={'text-align': 'center'}),

                html.Div([
                    html.Div([
                        html.Label("League Size"),
                        dcc.Input(id='league-size', type='number', min=4, max=12, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Your Pick"),
                        dcc.Input(id='user-pick', type='number', min=1, max=12, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("QB Slots"),
                        dcc.Input(id='qb-slots', type='number', min=1, max=2, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("RB Slots"),
                        dcc.Input(id='rb-slots', type='number', min=1, max=3, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("WR Slots"),
                        dcc.Input(id='wr-slots', type='number', min=1, max=3, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("TE Slots"),
                        dcc.Input(id='te-slots', type='number', min=1, max=2, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Flex W/R Slots"),
                        dcc.Input(id='flex-wr-slots', type='number', min=0, max=2, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Flex W/R/T Slots"),
                        dcc.Input(id='flex-wrt-slots', type='number', min=0, max=2, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Kicker Slots"),
                        dcc.Input(id='kicker-slots', type='number', min=0, max=1, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Defense Slots"),
                        dcc.Input(id='defense-slots', type='number', min=0, max=1, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                    
                    html.Div([
                        html.Label("Bench Slots"),
                        dcc.Input(id='bench-slots', type='number', min=0, max=10, style={'width': '100px'}),
                    ], style={'display': 'inline-block', 'margin-right': '10px'}),
                ], style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),
                
                html.Div([
                    html.Button("Start Draft", id='start-draft', n_clicks=0),
                    html.Button("Reset Draft", id='reset-draft', n_clicks=0),
                    html.Button("Draft Kicker", id='draft-kicker-button', n_clicks=0),
                    html.Button("Draft Defense", id='draft-defense-button', n_clicks=0),
                    ], style={'text-align': 'center', 'margin-top': '20px', 'magin-right': '2 px'}),
                
                html.Div([
                    dash_table.DataTable(
                        id='draft-board',
                        columns=[],  # Will be dynamically generated
                        data=[],  # Will be dynamically generated
                        style_table={'height': 'auto', 'overflowY': 'auto', 'margin-top': '2vh'},
                        style_cell={'textAlign': 'center', 'minWidth': '80px', 'width': '80px', 'maxWidth': '80px'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                    )
                ]),
                
                html.Div(id='recommended-players-container', style={'display': 'flex', 'justify-content': 'space-around', 'margin-top': '20px'}, children=[
                    html.Div(id='recommended-player-one', style={'width': '30%', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'text-align': 'center', 'cursor': 'pointer'}),
                    html.Div(id='recommended-player-two', style={'width': '30%', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'text-align': 'center', 'cursor': 'pointer'}),
                    html.Div(id='recommended-player-three', style={'width': '30%', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'text-align': 'center', 'cursor': 'pointer'}),
                ]),
            ]),
            ], style={'width': '68%', 'height': '80vh', 'float': 'left', 'box-sizing': 'border-box'}),  # Shared space for profile and draft board

    ], style={'display': 'flex', 'width': '100%', 'box-sizing': 'border-box'}),
])

# Callback to detect typing in the search bar
@app.callback(
    Output('is-typing', 'data'),
    Input('search-bar', 'value')
)
def update_typing_state(search_value):
    # If there is text in the search bar, set is_typing to True
    return True if search_value else False

@app.callback(
    [Output('league-size', 'value', allow_duplicate=True),
     Output('user-pick', 'value', allow_duplicate=True),
     Output('qb-slots', 'value', allow_duplicate=True),
     Output('rb-slots', 'value', allow_duplicate=True),
     Output('wr-slots', 'value', allow_duplicate=True),
     Output('te-slots', 'value', allow_duplicate=True),
     Output('flex-wr-slots', 'value', allow_duplicate=True),
     Output('flex-wrt-slots', 'value', allow_duplicate=True),
     Output('kicker-slots', 'value', allow_duplicate=True),
     Output('defense-slots', 'value', allow_duplicate=True),
     Output('bench-slots', 'value', allow_duplicate=True),
     Output('start-draft', 'n_clicks', allow_duplicate=True),
     Output('total-value-points', 'data', allow_duplicate=True),
     Output('total-slots', 'data', allow_duplicate=True),
     Output('draft-board-data', 'data', allow_duplicate=True),
     Output('selected-players', 'data', allow_duplicate=True),
     Output('user-drafted-players', 'data', allow_duplicate=True),
     Output('drafted-players', 'data', allow_duplicate=True),
     Output('previous-draft-board', 'data', allow_duplicate=True)],
    [Input('reset-draft', 'n_clicks')],
    [State('total-value-points', 'data'),
     State('total-slots', 'data'),
     State('draft-board-data', 'data'),
     State('selected-players', 'data'),
     State('user-drafted-players', 'data'),
     State('drafted-players', 'data'),
     State('previous-draft-board', 'data')],
    prevent_initial_call=True
)
def reset_draft(reset_n_clicks, total_value_points, total_slots, draft_board_data, selected_players, user_drafted_players, drafted_players, previous_draft_board):
    if reset_n_clicks > 0:
        # Reset all input fields to None or default values
        return [None, None, None, None, None, None, None, None, None, None, None, 0, {}, {}, [], [], [], [], []]
    
    # If the reset button was not pressed, return no update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('total-value-points', 'data'),
     Output('total-slots', 'data')],
    [Input('start-draft', 'n_clicks')],
    [State('qb-slots', 'value'),
     State('rb-slots', 'value'),
     State('wr-slots', 'value'),
     State('te-slots', 'value'),
     State('flex-wr-slots', 'value'),
     State('flex-wrt-slots', 'value'),
     State('kicker-slots', 'value'),
     State('defense-slots', 'value'),
     State('bench-slots', 'value'),
     ]
)
def calculate_and_assign_value_points(start_n_clicks, qb_slots, rb_slots, wr_slots, te_slots, flex_wr_slots, flex_wrt_slots, kicker_slots, defense_slots, bench_slots):
    if start_n_clicks > 0:
        def diminishing_value(slots, base_value):
            return sum(base_value / (1.25 ** i) for i in range(slots))

        # Calculate value points for each position
        qb_value_points = diminishing_value(qb_slots or 0, 100)
        rb_value_points = diminishing_value(rb_slots or 0, 100)
        wr_value_points = diminishing_value(wr_slots or 0, 100)
        te_value_points = diminishing_value(te_slots or 0, 100)
        flex_wr_value_points = diminishing_value(flex_wr_slots or 0, 100)
        flex_wrt_value_points = diminishing_value(flex_wrt_slots or 0, 100)
        kicker_value_points = 0
        defense_value_points = 0

        total_value_points = {
            'QB': qb_value_points,
            'RB': rb_value_points,
            'WR': wr_value_points,
            'TE': te_value_points,
            'FLEX_WR': flex_wr_value_points,
            'FLEX_WRT': flex_wrt_value_points,
            'K': kicker_value_points,
            'DEF': defense_value_points,
            'Bench': 0
        }
        
        total_slots = {
            'QB': qb_slots,
            'RB': rb_slots,
            'WR': wr_slots,
            'TE': te_slots,
            'FLEX_WR': flex_wr_slots,
            'FLEX_WRT': flex_wrt_slots,
            'K': kicker_slots,
            'DEF': defense_slots,
            'Bench': bench_slots
        }
        
        # Filter out positions with slots equal to 0
        total_value_points = {pos: value for pos, value in total_value_points.items() if total_slots[pos] > 0}
        total_slots = {pos: slots for pos, slots in total_slots.items() if slots > 0}
        
        return total_value_points, total_slots
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('draft-board', 'columns'),
     Output('draft-board', 'data'),
     Output('draft-board-data', 'data'),
     Output('selected-players', 'data')],
    [Input('start-draft', 'n_clicks'),
     Input({'type': 'draft-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input('draft-kicker-button', 'n_clicks'),
     Input('draft-defense-button', 'n_clicks')],
    [State('league-size', 'value'),
     State('user-pick', 'value'),
     State('qb-slots', 'value'),
     State('rb-slots', 'value'),
     State('wr-slots', 'value'),
     State('te-slots', 'value'),
     State('flex-wr-slots', 'value'),
     State('flex-wrt-slots', 'value'),
     State('kicker-slots', 'value'),
     State('defense-slots', 'value'),
     State('bench-slots', 'value'),
     State('draft-board-data', 'data'),
     State('selected-players', 'data'),
     State('all_pos_df', 'data')]
)
def handle_draft_and_create_board(
        start_n_clicks, draft_n_clicks, draft_kicker_n_clicks, draft_defense_n_clicks,
        league_size, user_pick, qb_slots, rb_slots, wr_slots, te_slots, flex_wr_slots, flex_wrt_slots, kicker_slots, defense_slots, bench_slots,
        draft_board_data, selected_players, all_pos_df):

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, selected_players

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Create draft board
    if triggered_id == 'start-draft' and start_n_clicks > 0:
        columns = [{'name': f'Team {i+1}', 'id': f'team_{i+1}'} for i in range(league_size)]
        if user_pick and user_pick <= league_size:
            columns[user_pick - 1] = {'name': 'User Team', 'id': f'team_{user_pick}'}

        total_slots = qb_slots + rb_slots + wr_slots + te_slots + flex_wr_slots + flex_wrt_slots + kicker_slots + defense_slots + bench_slots
        data = []
        pick_number = 1
        for row in range(total_slots):
            row_data = {}
            for col in range(league_size):
                actual_col = col if row % 2 == 0 else league_size - col - 1
                row_data[f'team_{actual_col + 1}'] = pick_number
                pick_number += 1
            data.append(row_data)
        
        draft_board_data = data  # Set draft_board_data with new data
        
        return columns, data, draft_board_data, selected_players

    # Handle draft kicker button press
    if triggered_id == 'draft-kicker-button':
        for row in draft_board_data:
            for team in row.keys():
                if isinstance(row[team], int) or not row[team]:  # Find an available pick slot
                    row[team] = "Kicker"  # Insert 'Kicker'
                    break
            else:
                continue
            break

    # Handle draft defense button press
    elif triggered_id == 'draft-defense-button':
        for row in draft_board_data:
            for team in row.keys():
                if isinstance(row[team], int) or not row[team]:  # Find an available pick slot
                    row[team] = "Defense"  # Insert 'Defense'
                    break
            else:
                continue
            break

    # Handle draft player button press
    elif 'draft-button' in triggered_id:
        if not draft_n_clicks or not draft_board_data:
            return dash.no_update, dash.no_update, dash.no_update, selected_players

        # Extract the player ID from the triggered component ID
        button_id = eval(triggered_id)
        player_id = button_id['index']

        if player_id in selected_players:
            return dash.no_update, dash.no_update, dash.no_update, selected_players

        # Find player information from the DataFrame
        all_pos_df = pd.DataFrame(all_pos_df)
        player_info = all_pos_df.loc[all_pos_df['id'] == player_id].iloc[0]
        player_name = player_info['name']
        position = player_info['position']

        selected_players.append(player_id)

        for row in draft_board_data:
            for team in row.keys():
                if isinstance(row[team], int) or not row[team]:  # Find an available pick slot
                    row[team] = f"{player_name}, {position}"  # Insert player name and position
                    break
            else:
                continue
            break

    return dash.no_update, draft_board_data, draft_board_data, selected_players

@app.callback(
    [Output('user-drafted-players', 'data'),
     Output('drafted-players', 'data'),
     Output('previous-draft-board', 'data'),
     Output('total-value-points', 'data', allow_duplicate=True),
     Output('total-slots', 'data', allow_duplicate=True),
     Output('all_pos_df', 'data', allow_duplicate=True)],
    [Input('draft-board', 'data')],
    [State('total-value-points', 'data'),
     State('total-slots', 'data'),
     State('all_pos_df', 'data'),
     State('user-drafted-players', 'data'),
     State('drafted-players', 'data'),
     State('previous-draft-board', 'data'),
     State('user-pick', 'value')],
    prevent_initial_call=True
)
def track_and_update_drafted_players(draft_board_data, total_value_points, total_slots, all_pos_df, user_drafted_players, drafted_players, previous_draft_board, user_pick):
    all_pos_df = pd.DataFrame(all_pos_df)
    all_pos_df['bonus_value_points'] = 0

    # Ensure user_drafted_players and drafted_players are lists
    user_drafted_players = user_drafted_players or []
    drafted_players = drafted_players or []

    # If there is no previous draft board, set it to an empty list
    previous_draft_board = previous_draft_board or []

    # Identify the user's team key based on the user_pick value
    user_team_key = f'team_{user_pick}'

    # Determine the last drafted player by comparing the current and previous draft boards
    last_drafted = None
    for current_row, previous_row in zip(draft_board_data, previous_draft_board):
        for key in current_row.keys():
            if current_row[key] != previous_row.get(key):
                last_drafted = current_row[key]
                break
        if last_drafted:
            break

    if last_drafted:
        # If it's a Kicker or Defense, handle accordingly
        if last_drafted in ["Kicker", "Defense"]:
            if last_drafted == "Kicker" and total_slots.get('K', 0) > 0:
                total_slots['K'] -= 1
            elif last_drafted == "Defense" and total_slots.get('DEF', 0) > 0:
                total_slots['DEF'] -= 1
        elif isinstance(last_drafted, str) and ', ' in last_drafted:
            player_name, position = last_drafted.split(', ')
            player_info = all_pos_df[(all_pos_df['name'] == player_name) & (all_pos_df['position'] == position)].iloc[0]

            if player_info['id'] not in drafted_players:
                drafted_players.append(player_info['id'])

                # Check if the last pick was made by the user's team
                if last_drafted == current_row.get(user_team_key):
                    user_drafted_players.append(player_info['id'])
                    used_slot, total_slots = subtract_slot(position, total_slots)
                    total_value_points = subtract_total_points(used_slot, player_info['value_points'], total_value_points)

                    positions_of_need_slot = [pos for pos, slots in total_slots.items() if slots == max(total_slots.values())]
                    if set(positions_of_need_slot) <= {'K', 'DEF'}:
                        positions_of_need_points = []
                    else:
                        positions_of_need_points = [pos for pos, points in total_value_points.items() if points == max(total_value_points.values())]
                        if 'FLEX_WR' in positions_of_need_points:
                            positions_of_need_points.extend(pos for pos in ['RB', 'WR'] if pos not in positions_of_need_points)
                        if 'FLEX_WRT' in positions_of_need_points:
                            positions_of_need_points.extend(pos for pos in ['WR', 'RB', 'TE'] if pos not in positions_of_need_points)

                    for _, player_info in all_pos_df.iterrows():
                        player_id = player_info['id']
                        player_position = player_info['position']
                        player_team = player_info['team']

                        if player_position in positions_of_need_points:
                            all_pos_df.loc[all_pos_df['id'] == player_id, 'bonus_value_points'] += 2

                        same_team_qbs = [
                            drafted_player for drafted_player in user_drafted_players
                            if all_pos_df.loc[all_pos_df['id'] == drafted_player, 'position'].iloc[0] == 'QB' and
                            all_pos_df.loc[all_pos_df['id'] == drafted_player, 'team'].iloc[0] == player_team
                        ]
                        if player_position in ['WR', 'TE'] and same_team_qbs:
                            all_pos_df.loc[all_pos_df['id'] == player_id, 'bonus_value_points'] += 3

                        same_team_pass_catchers = [
                            drafted_player for drafted_player in user_drafted_players
                            if all_pos_df.loc[all_pos_df['id'] == drafted_player, 'position'].iloc[0] in ['WR', 'TE'] and
                            all_pos_df.loc[all_pos_df['id'] == drafted_player, 'team'].iloc[0] == player_team
                        ]
                        if player_position == 'QB' and same_team_pass_catchers:
                            all_pos_df.loc[all_pos_df['id'] == player_id, 'bonus_value_points'] += 3

                        same_team_rbs = [
                            drafted_player for drafted_player in user_drafted_players
                            if all_pos_df.loc[all_pos_df['id'] == drafted_player, 'position'].iloc[0] == 'RB' and
                            all_pos_df.loc[all_pos_df['id'] == drafted_player, 'team'].iloc[0] == player_team
                        ]
                        if player_position in ['WR', 'TE', 'QB'] and same_team_rbs:
                            all_pos_df.loc[all_pos_df['id'] == player_id, 'bonus_value_points'] -= 5

                        same_team_qbs_wr_te = [
                            drafted_player for drafted_player in user_drafted_players
                            if all_pos_df.loc[all_pos_df['id'] == drafted_player, 'position'].iloc[0] in ['QB', 'WR', 'TE'] and
                            all_pos_df.loc[all_pos_df['id'] == drafted_player, 'team'].iloc[0] == player_team
                        ]
                        if player_position == 'RB' and same_team_qbs_wr_te:
                            all_pos_df.loc[all_pos_df['id'] == player_id, 'bonus_value_points'] -= 5

        # If the pick was not made by the user's team, skip the bonus points logic
        else:
            drafted_players.append(player_info['id'])

        print(total_value_points, total_slots)

    return user_drafted_players, drafted_players, draft_board_data, total_value_points, total_slots, all_pos_df.to_dict('records')

@app.callback(
    [Output('recommended-player-one', 'children'),
     Output('recommended-player-two', 'children'),
     Output('recommended-player-three', 'children')],
    [Input('start-draft', 'n_clicks'),
     Input({'type': 'draft-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input('user-drafted-players', 'data'),
     Input('drafted-players', 'data')],
    [State('all_pos_df', 'data')]
)
def recommend_players(start_n_clicks, draft_n_clicks, user_drafted_players, drafted_players, all_pos_df):
    
    if start_n_clicks == 0:
        return [html.Div("No recommendations available")] * 3

    ctx = dash.callback_context
    
    if ctx.triggered and draft_n_clicks:
    
        all_pos_df = pd.DataFrame(all_pos_df)
        if 'bonus_value_points' not in all_pos_df.columns:
            all_pos_df['bonus_value_points'] = 0
        available_players = all_pos_df[(~all_pos_df['id'].isin(user_drafted_players)) & (~all_pos_df['id'].isin(drafted_players))]
        available_players['total_value_points'] = available_players['value_points'] + available_players['bonus_value_points']
        available_players = available_players.sort_values(by='total_value_points', ascending=False)
        player_recommendations = available_players.head(3)['id'].tolist()
        
        top_players = all_pos_df[all_pos_df['id'].isin(player_recommendations[:3])]

        recommendations = []
        
        def format_team_change(team_change_value):
                return "Yes" if team_change_value == 1 else "No"
        
        for idx, player_info in top_players.iterrows():
            player_id = player_info['id']
            player_name = player_info['name']
            position = player_info['Depth']
            team_abbr = player_info['team']
            age = player_info['age']
            exp = player_info['years_exp']
            tm_change = format_team_change(player_info['team_change'])
            if len(all_season[all_season['id'] == player_id]) > 0:
                player_seasons = all_season[all_season['id'] == player_id]
                avg_rank = player_seasons['player_rank'].mean().round(2)
                worst_rank = player_seasons['player_rank'].max()
                best_rank = player_seasons['player_rank'].min()
            else:
                avg_rank = best_rank = worst_rank = 'N/A'
            adp = player_info['ADP']
            adp_rank = player_info['ADP_rank']
            proj_rank = player_info['predicted_rank']
            proj_points = player_info['predicted_points']
            
            team_logo_url = player_info['team_logo_espn']
            headshot_url = player_info['headshot_url'] if player_info['headshot_url'] != 0 else team_logo_url

            # Building the recommended player content
            player_content = html.Div([
                html.Div([
                    html.Img(src=headshot_url, style={'width': '150px', 'border-radius': '50%', 'background': f"url({team_logo_url})", 'background-size': 'cover', 'float': 'left', 'margin-right': '10px'}),
                    html.H4(f"{player_name}", style={'margin': '0', 'font-size': '2vh', 'font-weight': 'bold', 'font-family': 'monospace', 'display': 'inline-block', 'vertical-align': 'middle'})
                ], style={'display': 'flex', 'align-items': 'center'}),
                html.Hr(style={'margin': '1vh 0'}),
                html.Div([
                    html.Div([
                        html.Span("Team", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{team_abbr}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Position", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{position}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Age", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{age}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Exp", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{exp}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '1vh'}),
                html.Div([
                    html.Div([
                        html.Span("ADP", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{adp}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("ADP_Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{adp_rank}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Proj Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{proj_rank}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Proj Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{proj_points}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '1vh'}),

                html.Div([
                    html.Div([
                        html.Span("Avg Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{avg_rank}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Best Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{best_rank}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Best Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{worst_rank}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                    html.Div([
                        html.Span("Team Change", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                        html.Br(),
                        html.Span(f"{tm_change}", style={'font-family': 'monospace'})
                    ], style={'width': '25%', 'text-align': 'center'}),
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '1vh'}),
            ], id=f'recommended-player-{idx}', style={'cursor': 'pointer', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'box-shadow': '0 0.2vh 0.4vh rgba(0, 0, 0, 0.05)'})

            recommendations.append(player_content)

        while len(recommendations) < 3:
            recommendations.append(html.Div("No further recommendations"))

        return recommendations

@app.callback(
    [Output('profile-container', 'style'),
     Output('draft-board-container', 'style'),
     Output('toggle-profile', 'children')],
    [Input('toggle-profile', 'n_clicks'),
     Input('start-draft', 'n_clicks'),
     Input('draft-board-data', 'data')],
    [State('profile-container', 'style'),
     State('draft-board-container', 'style')]
)
def toggle_profile(toggle_n_clicks, start_n_clicks, draft_board_data, profile_style, draft_board_style):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'start-draft' and start_n_clicks > 0:
        return {'display': 'none'}, {'display': 'block'}, "Show Profile"

    if toggle_n_clicks % 2 == 1:
        return {'display': 'none'}, {'display': 'block'}, "Show Profile"
    else:
        return {'display': 'block'}, {'display': 'none'}, "Hide Profile"

@app.callback(
    Output('start-draft', 'disabled'),
    Output('draft-enabled', 'data'),
    [Input('league-size', 'value'),
     Input('user-pick', 'value'),
     Input('qb-slots', 'value'),
     Input('rb-slots', 'value'),
     Input('wr-slots', 'value'),
     Input('te-slots', 'value'),
     Input('flex-wr-slots', 'value'),
     Input('flex-wrt-slots', 'value'),
     Input('kicker-slots', 'value'),
     Input('defense-slots', 'value'),
     Input('bench-slots', 'value')]
)
def enable_start_draft(league_size, user_pick, qb_slots, rb_slots, wr_slots, te_slots, flex_wr_slots, flex_wrt_slots, kicker_slots, defense_slots, bench_slots):
    if all(v is not None for v in [league_size, user_pick, qb_slots, rb_slots, wr_slots, te_slots, flex_wr_slots, flex_wrt_slots, kicker_slots, defense_slots, bench_slots]):
        return False, True
    return True, False

@app.callback(
    Output('player-list', 'children'),
    [Input('sort-radio', 'value'),
     Input('search-bar', 'value'),
     Input('selected-players', 'data'),
     Input('draft-enabled', 'data'),
     Input('is-typing', 'data')],
    prevent_initial_call=True
)
def update_player_list(sort_by, search_value, selected_players, draft_enabled, is_typing):
    df = pd.DataFrame(all_pos_df)

    # Sort logic
    if sort_by == 'ADP':
        df = df.sort_values(by='ADP')  # Smallest to largest
    elif sort_by == 'Draft Value':
        df = df.sort_values(by='value_points', ascending=False)  # Largest to smallest

    # Filter by search value
    if search_value:
        df = df[df['name'].str.contains(search_value, case=False, na=False)]

    # Disable draft buttons if typing or if drafting is not enabled
    draft_button_disabled = is_typing or not draft_enabled

    # Filter out drafted players
    df = df[~df['id'].isin(selected_players)]

    player_list = []
    for i, row in df.iterrows():
        player_name_link = html.A(
            row['name'],
            href="#",  # Replace with actual player profile URL if available
            id={'type': 'player-name', 'index': row['id']},  # Use player ID as the index
            style={'width': '39%', 'display': 'inline-block', 'cursor': 'pointer'}
        )

        player_list.append(html.Li([
            player_name_link,
            html.Span(row['team'], style={'width': '15%', 'display': 'inline-block'}),
            html.Span(row['ADP'], style={'width': '15%', 'display': 'inline-block'}),
            html.Span(row['predicted_rank'], style={'width': '15%', 'display': 'inline-block'}),
            html.Button('Draft', id={'type': 'draft-button', 'index': row['id']}, 
                        disabled=draft_button_disabled, style={'width': '15%', 'display': 'inline-block'})
        ], style={
            'display': 'flex', 
            'padding': '2%', 
            'border-bottom': '0.3vh solid #ddd',
            'box-sizing': 'border-box',
            'width': '100%'
        }))

    return player_list

@app.callback(
    [Output('selected-player-index', 'data'),
     Output('clicked-player-index', 'data'),
     Output('player-details', 'children'),
     Output('headshot-container', 'children'),
     Output('headshot-container', 'style'),
     Output('team-details', 'children'),
     Output('pros-cons-container', 'children'),
     Output('projections-container', 'children')],
    [Input({'type': 'player-name', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('sort-radio', 'value'),
     State('search-bar', 'value'),
     State('clicked-player-index', 'data')]
)
def update_player_selection_and_details(n_clicks, sort_by, search_value, clicked_index):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Return the expected 8 outputs with default values
        return (clicked_index, None, 
                ["Player not found.", "", {}, "No team data available.", "No pros/cons available.", "No projections available.", None], 
                None, 
                {}, 
                "No team data available.", 
                "No pros/cons available.", 
                "No projections available.")

    # Determine which player name was clicked
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    player_id = eval(triggered_id)['index']  # Extract the player ID

    df = all_pos_df.sort_values(by=sort_by)
    if search_value:
        df = df[df['name'].str.contains(search_value, case=False, na=False)]

    # Find the correct player row by ID using loc instead of iloc
    player_info = df.loc[df['id'] == player_id].iloc[0] if not df[df['id'] == player_id].empty else None
    
    if player_info is None:
        # Return the expected 8 outputs with default values if player is not found
        return (None, None, 
                ["Player not found.", "", {}, "No team data available.", "No pros/cons available.", "No projections available.", None], 
                None, 
                {}, 
                "No team data available.", 
                "No pros/cons available.", 
                "No projections available.")

    # Update clicked_index to be the player_id instead of DataFrame index
    clicked_index = player_id

    # Further profile details fetching logic remains unchanged
    position = player_info['position']
    season_df = season_data[position]
    player_seasons = season_df[season_df['id'] == player_id]

    if not player_seasons.empty:
        filtered_df = player_seasons[player_seasons['Depth'].isin(['QB1', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE1'])]
        availability = filtered_df['games'].mean() if len(filtered_df) > 0 else 'N/A'
        avg_rank = player_seasons['player_rank'].mean().round(2)
        worst_rank = player_seasons['player_rank'].max()
        best_rank = player_seasons['player_rank'].min()
        best_weekly_ppr_points = player_seasons['weekly_ppr_points'].max()
        explosiveness, explosiveness_threshold = calculate_explosiveness(player_id, position)
        production = calculate_and_color_production(player_id, position)
        consistency_grade = calculate_consistency(player_id, position)
    else:
        avg_rank = best_rank = worst_rank = best_weekly_ppr_points = consistency_grade = availability = explosiveness = explosiveness_threshold = "N/A"
        production = '#d3d3d3'

    team_abbr = player_info['team']
    team_logo_url = player_info['team_logo_espn']
    headshot_url = player_info['headshot_url'] if player_info['headshot_url'] != 0 else team_logo_url
    headshot_style = {'width': '75%', 'height': '75%', 'object-fit': 'contain', 'border-radius': '0.5vh'} if player_info['headshot_url'] != 0 else {'width': '60%', 'height': '60%', 'object-fit': 'contain', 'border-radius': '0.5vh'}

    team_adps = {
        'QB1': qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB1')]['ADP'].values[0] if len(qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB1')]) > 0 else 999,
        'QB2': qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB2')]['ADP'].values[0] if len(qb_df[(qb_df['team'] == team_abbr) & (qb_df['Depth'] == 'QB2')]) > 0 else 999,
        'RB1': rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB1')]['ADP'].values[0] if len(rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB1')]) > 0 else 999,
        'RB2': rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB2')]['ADP'].values[0] if len(rb_df[(rb_df['team'] == team_abbr) & (rb_df['Depth'] == 'RB2')]) > 0 else 999,
        'WR1': wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR1')]['ADP'].values[0] if len(wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR1')]) > 0 else 999,
        'WR2': wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR2')]['ADP'].values[0] if len(wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR2')]) > 0 else 999,
        'WR3': wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR3')]['ADP'].values[0] if len(wr_df[(wr_df['team'] == team_abbr) & (wr_df['Depth'] == 'WR3')]) > 0 else 999,
        'TE1': te_df[(te_df['team'] == team_abbr) & (te_df['Depth'] == 'TE1')]['ADP'].values[0] if len(te_df[(te_df['team'] == team_abbr) & (te_df['Depth'] == 'TE1')]) > 0 else 999,
    }

    def convert_height(height_in_inches):
        feet = height_in_inches // 12
        inches = height_in_inches % 12
        return f"{feet}'{inches}\""

    def format_team_change(team_change_value):
        return "Yes" if team_change_value == 1 else "No"

    player_details = html.Div([
        html.H4(f"{player_info['name']}", style={'margin': '0', 'font-size': '2vh', 'font-weight': 'bold', 'font-family': 'monospace'}),
        html.Hr(style={'margin': '1vh 0'}),
        html.Div([
            html.Div([
                html.Span("Position", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['Depth']}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
            html.Div([
                html.Span("Age", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['age']}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
            html.Div([
                html.Span("Experience", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['years_exp']} yrs", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '1vh'}),

        html.Div([
            html.Div([
                html.Span("Height", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{convert_height(player_info['height'])}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
            html.Div([
                html.Span("Weight", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['weight']} lbs", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
            html.Div([
                html.Span("Team Change", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{format_team_change(player_info['team_change'])}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        html.Div([
            html.Div([
                html.Span("Avg Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{avg_rank}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
            html.Div([
                html.Span("Best Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{best_rank}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
            html.Div([
                html.Span("Worst Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{worst_rank}", style={'font-family': 'monospace'})
            ], style={'width': '30%', 'text-align': 'center'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '1vh'}),
    ])

    headshot_content = html.Div([
        html.Img(src=headshot_url, style=headshot_style),
        html.Img(src=team_logo_url, style={'position': 'absolute', 'top': '5%', 'right': '5%', 'width': '25%', 'height': 'auto'})
    ])

    headshot_container_style = {
        'position': 'relative',
        'width': '19.8%',  
        'height': '20%',  
        'border': '0.3vh solid #ddd',
        'border-radius': '0.5vh',
        'box-shadow': '0 0.2vh 0.4vh rgba(0, 0, 0, 0.1)',
        'margin-bottom': '1%',
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'box-sizing': 'border-box',
        'background-color': '#fff',  
        'overflow': 'hidden'  
    }

    def find_starter_with_details(df, team_abbr, depth):
        starter = df[(df['team'] == team_abbr) & (df['Depth'] == depth)]
        if not starter.empty:
            player_name = starter['name'].iloc[0]
            adp = starter['ADP'].iloc[0].round()
            exp = starter['years_exp'].iloc[0]
            proj_points = starter['predicted_points'].iloc[0]
            return [depth, player_name, adp, exp, proj_points]
        else:
            return [depth, "N/A", "N/A", "N/A", "N/A"]

    qb1 = find_starter_with_details(qb_df, team_abbr, 'QB1')
    rb1 = find_starter_with_details(rb_df, team_abbr, 'RB1')
    rb2 = find_starter_with_details(rb_df, team_abbr, 'RB2')
    wr1 = find_starter_with_details(wr_df, team_abbr, 'WR1')
    wr2 = find_starter_with_details(wr_df, team_abbr, 'WR2')
    wr3 = find_starter_with_details(wr_df, team_abbr, 'WR3')
    te1 = find_starter_with_details(te_df, team_abbr, 'TE1')
    starters = [qb1, rb1, rb2, wr1, wr2, wr3, te1]

    if player_info['years_exp'] == 0:
        team_stats = get_team_stats_from_non_rookie(team_abbr)
        prev_wins = team_stats['prev_wins']
        prev_team_rank = team_stats['prev_team_rank']
        prev_pass_rank = team_stats['prev_pass_rank']
        prev_rush_rank = team_stats['prev_rush_rank']
    else:
        prev_wins = player_info['prev_wins']
        prev_team_rank = player_info['prev_team_rank']
        prev_pass_rank = player_info['prev_pass_rank']
        prev_rush_rank = player_info['prev_rush_rank']
    
    team_details = html.Div([
        html.H4(f"{player_info['team_name']}", style={'margin': '0', 'font-size': '2.5vh', 'font-weight': 'bold', 'font-family': 'monospace'}),
        html.Hr(style={'margin': '1vh 0'}),
        html.Div([
            html.Div([
                html.Span("Division", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['team_division']}", style={'font-family': 'monospace'})
            ], style={'width': '33%', 'text-align': 'center'}),
            html.Div([
                html.Span("Favored Matchups", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['projected_wins']}", style={'font-family': 'monospace'})
            ], style={'width': '33%', 'text-align': 'center'}),
            html.Div([
                html.Span("Primetime", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{player_info['primetime']}", style={'font-family': 'monospace'})
            ], style={'width': '33%', 'text-align': 'center'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '1vh'}),

        html.Div([
            html.Div([
                html.Span("Prev Team Wins", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{prev_wins}", style={'font-family': 'monospace'})
            ], style={'width': '25%', 'text-align': 'center'}),
            html.Div([
                html.Span("Prev Team Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{prev_team_rank}", style={'font-family': 'monospace'})
            ], style={'width': '25%', 'text-align': 'center'}),
            html.Div([
                html.Span("Prev Pass Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{prev_pass_rank}", style={'font-family': 'monospace'})
            ], style={'width': '25%', 'text-align': 'center'}),
            html.Div([
                html.Span("Prev Rush Rank", style={'font-weight': 'bold', 'font-family': 'monospace'}),
                html.Br(),
                html.Span(f"{prev_rush_rank}", style={'font-family': 'monospace'})
            ], style={'width': '25%', 'text-align': 'center'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        html.Hr(style={'margin': '1vh 0'}),

        html.Div([
            html.Div([
                html.Span("Pos", style={'width': '10%', 'font-weight': 'bold', 'text-align': 'left', 'font-family': 'monospace'}),
                html.Span("Name", style={'width': '50%', 'font-weight': 'bold', 'text-align': 'left', 'font-family': 'monospace'}),
                html.Span("ADP", style={'width': '15%', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'monospace'}),
                html.Span("Exp", style={'width': '15%', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'monospace'}),
                html.Span("Proj", style={'width': '10%', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'monospace'}),
            ], style={'display': 'flex', 'padding': '0.5vh 0'}),

            *[html.Div([
                html.Span(starter[0], style={'width': '10%', 'text-align': 'left', 'font-family': 'monospace'}),
                html.Span(starter[1], style={'width': '50%', 'text-align': 'left', 'font-family': 'monospace'}),
                html.Span(starter[2], style={'width': '15%', 'text-align': 'center', 'font-family': 'monospace'}),
                html.Span(starter[3], style={'width': '15%', 'text-align': 'center', 'font-family': 'monospace'}),
                html.Span(starter[4], style={'width': '10%', 'text-align': 'center', 'font-family': 'monospace'}),
            ], style={'display': 'flex', 'padding': '0.5vh 0'}) for starter in starters]
        ])
    ])

    pros_cons_items = [
        "Experience", "Best Season", "Team Proj", "Avg Weekly Points", "O-Line", "Consistency",
        "Explosiveness", "Production", "Competition", 'Availability'
    ]

    pros_cons_content = html.Div([
        html.Div(
            pros_cons_items[i], 
            style={
                'background-color': color,  
                'width': '48%',  
                'height': '6.4vh',  
                'margin': '1%',  
                'display': 'flex', 
                'align-items': 'center', 
                'justify-content': 'center',
                'text-align': 'center', 
                'font-family': 'monospace',
                'font-weight': 'bold',
                'border-radius': '0.5vh',
                'box-shadow': '0 0.2vh 0.4vh rgba(0, 0, 0, 0.1)'
            }
        ) for i, color in enumerate([
            determine_experience_color(position, player_info['years_exp']),
            determine_best_season_color(position, best_rank),
            determine_color(player_info['projected_wins'], (5, 9, 10)),
            determine_avg_weekly_points_color(position, best_weekly_ppr_points),
            determine_color(player_info['line_rating'], (22, 21, 10), True),
            determine_color(consistency_grade, (0.29, 0.69, 0.7)),
            determine_color(explosiveness, explosiveness_threshold),  
            production,
            determine_competition_color(position, player_info['Depth'], team_adps),
            determine_color(availability, (11, 14, 15))
        ])
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between', 'width': '100%'})
    
    upside, risk = calculate_gauges(position, player_info, best_rank, best_weekly_ppr_points, consistency_grade, team_adps, availability, explosiveness, production)
    
    projections_content = html.Div([
        html.Div([
            daq.Gauge(
                label="Upside",
                value=upside,
                min=0,
                max=10,
                showCurrentValue=True,
                color={"gradient": True, "ranges": {"green": [0, 10]}},
                size=200,
                scale={'start': 0, 'interval': 1, 'labelInterval': 2, 'tickwidth': 2}
            ),
            daq.Gauge(
                label="Risk",
                value=risk,
                min=0,
                max=10,
                showCurrentValue=True,
                color={"gradient": True, "ranges": {"red": [0, 10]}},
                size=200,
                scale={'start': 0, 'interval': 1, 'labelInterval': 2, 'tickwidth': 2}
            ),
        ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}),
        html.Hr(style={'margin': '1vh 0'}),
        
        html.Div([
            html.Div([
                html.Span(f"ADP Rank", style={'width': '33%', 'text-align': 'center',}),
                html.Span(f"Projected Rank", style={'width': '33%','text-align': 'center',}),
                html.Span(f"Projected Points", style={'width': '33%','text-align': 'center', }),
        ], style={'display': 'flex', 'padding': '0.5vh 0', 'margin-top': '2vh'}),
        
            html.Div([
                html.Span(f"{player_info['ADP_rank']}", style={'width': '33%','text-align': 'center',}),
                html.Span(f"{player_info['predicted_rank']}", style={'width': '33%','text-align': 'center',}),
                html.Span(f"{player_info['predicted_points']}", style={'width': '33%', 'text-align': 'center',}),
        ], style={'display': 'flex', 'padding': '0.5vh 0', 'margin-top': '2vh'}),
    ], style={'width': '100%', 'height': '100%', 'box-sizing': 'border-box'})
        ])
    
    return player_id, player_id, player_details, headshot_content, headshot_container_style, team_details, pros_cons_content, projections_content
    
@app.callback(
    [Output('season-filter', 'options'),
     Output('season-points-graph', 'figure'),
     Output('season-filter', 'value'),
     Output('points-filter', 'style'),
     Output('clear-season-filter', 'n_clicks')],
    [Input('clicked-player-index', 'data'),
     Input('points-filter', 'value'),
     Input('season-filter', 'value'),
     Input('clear-season-filter', 'n_clicks')],
    [State('sort-radio', 'value')]
)
def update_season_options_and_graph(player_id, points_filter, selected_season, n_clicks_clear, sort_by):
    if n_clicks_clear > 0:
        selected_season = None

    if player_id is None:
        return [], go.Figure(), None, {'display': 'none'}, 0

    player_info = all_pos_df[all_pos_df['id'] == player_id].iloc[0]
    position = player_info['position']
    is_rookie = player_info['years_exp'] == 0

    team_color = player_info['team_color']
    team_color2 = player_info['team_color2']

    if is_rookie:
        fig = go.Figure()
        fig.update_layout(
            title=f"{player_info['name']} is a Rookie - No Previous Seasons Data",
            xaxis_title='Season',
            yaxis_title='Fantasy Points',
            title_font_size=24,  # Larger title font
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14)),
            margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins
            shapes=[
                dict(
                    type="line",
                    x0=0,
                    y0=15,
                    x1=1,
                    y1=15,
                    xref='paper',
                    yref='y',
                    line=dict(color=team_color2, width=2, dash='dash')
                )
            ]
        )
        return [], fig, None, {'display': 'none'}, 0

    filtered_season_df = season_data[position][season_data[position]['id'] == player_id]
    available_seasons = filtered_season_df['season'].sort_values().unique()
    season_options = [{'label': str(season), 'value': season} for season in available_seasons]

    filtered_weekly_df = weekly_df[weekly_df['id'] == player_id]
    all_seasons = pd.DataFrame({'season': range(2012, 2024)})

    if selected_season is None:
        if points_filter == 'weekly_ppr_points':
            season_points = filtered_season_df[['season', 'weekly_ppr_points']].groupby('season').mean().reset_index()
            season_points = pd.merge(all_seasons, season_points, on='season', how='left').fillna(0)
            fig = px.bar(season_points, x='season', y='weekly_ppr_points',
                         title=f"{player_info['name']}'s Weekly Avg Fantasy Points ({available_seasons[0]}-{available_seasons[-1]})",
                         labels={'season': 'Season', 'weekly_ppr_points': 'Weekly Avg Fantasy Points'},
                         template='plotly_white')

            fig.update_traces(marker_color=team_color)
            fig.add_hline(y=15, line_dash="dash", line_color=team_color2, annotation_text="15 Points Threshold", annotation_position="top left")

            fig.update_layout(
                xaxis_title='Season',
                yaxis_title='Weekly Fantasy Points',
                title_font_size=20,  # Larger title font
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12), gridcolor='lightgrey'),
                margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            player_seasons = filtered_weekly_df[['season', points_filter]].groupby('season').sum().reset_index()
            season_points = pd.merge(all_seasons, player_seasons, on='season', how='left').fillna(0)
            fig = px.bar(season_points, x='season', y=points_filter,
                         title=f"{player_info['name']}'s Total Fantasy Points ({available_seasons[0]}-{available_seasons[-1]})",
                         labels={'season': 'Season', points_filter: 'Fantasy Points'},
                         template='plotly_white')

            fig.update_traces(marker_color=team_color)
            fig.add_hline(y=250, line_dash="dash", line_color=team_color2, annotation_text="250 Points Threshold", annotation_position="top left")

            fig.update_layout(
                xaxis_title='Season',
                yaxis_title='Total Fantasy Points',
                title_font_size=20,  # Larger title font
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12), gridcolor='lightgrey'),
                margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
    else:
        weekly_points = filtered_weekly_df[filtered_weekly_df['season'] == selected_season]
        all_weeks = pd.DataFrame({'week': range(1, 18)})
        weekly_points = pd.merge(all_weeks, weekly_points[['week', 'ppr_points']], on='week', how='left').fillna(0)
        fig = px.scatter(weekly_points, x='week', y='ppr_points',
                         title=f"{player_info['name']}'s Weekly Fantasy Points ({selected_season})",
                         labels={'week': 'Week', 'ppr_points': 'Fantasy Points'},
                         template='plotly_white')

        fig.update_traces(mode='lines+markers', marker=dict(size=8, color=team_color))

        fig.add_hline(y=15, line_dash="dash", line_color=team_color2, annotation_text="15 Points Threshold", annotation_position="top left")

        fig.update_layout(
            xaxis_title='Week',
            yaxis_title='Weekly Fantasy Points',
            title_font_size=20,  # Larger title font
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12), gridcolor='lightgrey'),
            margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

    points_filter_style = {'display': 'block'} if selected_season is None else {'display': 'none'}
    return season_options, fig, selected_season, points_filter_style, 0

@app.callback(
    [Output('season-stats-content', 'children'),
     Output('season-slider', 'min'),
     Output('season-slider', 'max'),
     Output('season-slider', 'marks'),
     Output('season-slider', 'value')],
    Input('season-tabs', 'value'),
    Input('selected-player-index', 'data'),
    Input('season-slider', 'value'),
    State('sort-radio', 'value'),
    State('season-slider', 'min'),
    State('season-slider', 'max')
)
def update_season_stats(selected_tab, selected_index, slider_range, sort_by, current_min, current_max):
    if selected_index is None:
        return "Select a player to view previous season stats.", 2000, 2023, {2000: '2000', 2023: '2023'}, [2000, 2023]

    # Locate the player using loc instead of iloc
    player_info = all_pos_df.loc[all_pos_df['id'] == selected_index].iloc[0]
    player_id = player_info['id']
    position = player_info['position']

    season_df = season_data[position]

    player_seasons = season_df[season_df['id'] == player_id].sort_values(by='season', ascending=False)

    if player_seasons.empty:
        return "No previous season stats available for this player.", 2000, 2023, {2000: '2000', 2023: '2023'}, [2000, 2023]

    min_season = player_seasons['season'].min()
    max_season = player_seasons['season'].max()

    if current_min != min_season or current_max != max_season:
        slider_range = [min_season, max_season]

    filtered_seasons = player_seasons[(player_seasons['season'] >= slider_range[0]) & 
                                      (player_seasons['season'] <= slider_range[1])]

    marks = {year: str(year) for year in range(min_season, max_season + 1)}

    common_cols = ['season', 'team', 'games']
    common_header = ['SEA', 'TM', 'G']

    if selected_tab == 'General':
        cols = common_cols + ['starts', 'offense_snaps', 'Depth', 'player_rank', 'ppr_points', 'weekly_ppr_points', 'consistency_grade']
        header = common_header + ['Start', 'Snaps', 'Depth', 'Rank', 'PTS', 'WK PTS', 'Stability']
        filtered_seasons['ppr_points'] = filtered_seasons['ppr_points'].round()
        filtered_seasons['weekly_ppr_points'] = filtered_seasons['weekly_ppr_points'].round(2)
        
    elif selected_tab == 'Pass':
        cols = common_cols + ['completions', 'attempts', 'Cmp %', 'passing_yards', 'passing_tds', 'interceptions', 'sacks']
        header = common_header + ['CMP', 'ATT', 'CMP %', 'YRD', 'TD', 'INT', 'SACK']
        filtered_seasons['Cmp %'] = (filtered_seasons['completions'] / filtered_seasons['attempts']).round(2)
        filtered_seasons['Cmp %'].fillna(0, inplace=True)
    
    elif selected_tab == 'Rush':
        cols = common_cols + ['carries', 'rushing_yards', 'rush_ypa', 'rushing_tds', 'inside5_rush_pct', 'fumbles_lost']
        header = common_header + ['ATT', 'YRD', 'Y/A', 'TD', 'Goal Line %', 'FUM']
    elif selected_tab == 'Rec':
        cols = common_cols + ['targets', 'receptions', 'receiving_yards', 'rec_ypc', 'receiving_tds', 'inside20_target_pct']
        header = common_header + ['TGT', 'REC', 'YRD', 'Y/R', 'TD', 'RZ TGT %']

    header_row = html.Div([
        html.Span(header[i], style={'width': '8%', 'font-weight': 'bold', 'text-align': 'center'}) for i in range(len(common_header))
    ] + [
        html.Span(header[i], style={'width': f'{90/len(cols[len(common_header):])}%', 'font-weight': 'bold', 'text-align': 'center'}) for i in range(len(common_header), len(header))
    ], style={'display': 'flex', 'padding': '1%', 'border-bottom': '0.2vh solid #ddd', 'box-sizing': 'border-box', 'width': '100%'})

    season_rows = []
    for i in range(len(filtered_seasons)):
        row = []
        for j, col in enumerate(cols):
            value = filtered_seasons.iloc[i][col] if col in filtered_seasons.columns else 'N/A'
            if pd.isnull(value):
                value = 'N/A'
            width = '8%' if j < len(common_header) else f'{90/len(cols[len(common_header):])}%'
            row.append(html.Span(value, style={'width': width, 'display': 'inline-block', 'text-align': 'center', 'overflow': 'hidden', 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}))
        season_rows.append(html.Div(row, style={'display': 'flex', 'padding': '1%', 'border-bottom': '0.2vh solid #ddd', 'box-sizing': 'border-box', 'width': '100%'}))

    return [header_row] + season_rows, min_season, max_season, marks, slider_range
    
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    app.run_server(debug=True)
