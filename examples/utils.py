import pyro
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def run_inference(model, guide, game_info, obs, num_steps=2000, optimizer=None, lr=0.01, verbose=True, loss_log_interval=1):
    
    pyro.clear_param_store()

    if optimizer is None:
        optimizer = pyro.optim.Adam({"lr": lr})

    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    losses = []

    pbar = tqdm(range(num_steps))
    for step in pbar:
        loss = svi.step(game_info, obs)
        if step % loss_log_interval == 0 and verbose:
            pbar.set_description("Loss = %f" % loss)
        losses.append(loss)
    
    return losses


def load_data(data_path=''):
    # read and format data
    df = pd.read_csv(data_path)

    # make a list of unique coaches
    coaches1 = df['team1_coach_id']
    coaches2 = df['team2_coach_id']
    unique_coaches = pd.concat([coaches1, coaches2]).unique()
    unique_coaches.sort()

    num_coaches = len(unique_coaches)

    # prepare some vectors
    coach_winrates = np.zeros(num_coaches)
    coach_num_matches = np.zeros(num_coaches)
    coach_num_wins = np.zeros(num_coaches)
    coach_num_draws = np.zeros(num_coaches)

    # calculate winrates for each coach
    for id, coach_id in enumerate(unique_coaches):
        coach_num_matches[id]  = len(df[(df['team1_coach_id'] == coach_id) | (df['team2_coach_id'] == coach_id)])
        coach_num_wins[id] = df[(df['team1_coach_id'] == coach_id) & (df['team1_win'] == 1)].shape[0] + df[(df['team2_coach_id'] == coach_id) & (df['team2_win'] == 1)].shape[0]
        coach_num_draws[id] = df[(df['team1_coach_id'] == coach_id) & (df['team1_win'] == 0)].shape[0] + df[(df['team2_coach_id'] == coach_id) & (df['team2_win'] == 0)].shape[0]
        coach_winrates[id] = (coach_num_wins[id] + 0.5 * coach_num_draws[id]) / coach_num_matches[id]

    # sort the coaches by winrate 
    indices = np.lexsort((unique_coaches, coach_winrates, coach_num_matches))
    sorted_coaches = unique_coaches[indices[::-1]]
    sorted_winrates = coach_winrates[indices[::-1]]
    sorted_num_matches = coach_num_matches[indices[::-1]]

    # make a dictionary for the coaches matching index and coach_id
    coach_dict = {coach: i for i, coach in enumerate(sorted_coaches)}

    # make variables for the model sorted by winrate
    id1 = torch.tensor(coaches1.map(coach_dict).values).long()
    id2 = torch.tensor(coaches2.map(coach_dict).values).long()
    obs = torch.tensor(df['team1_win'].values)

    data = {
        'all_data': df,
        'coach1_id': id1,
        'coach2_id': id2,
        'result': obs,
        'sorted_coaches': sorted_coaches,
        'sorted_winrates': sorted_winrates,
        'sorted_num_matches': sorted_num_matches,
        'num_coaches': num_coaches,
    }

    return data




