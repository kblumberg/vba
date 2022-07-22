import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV

base_dir = ''

##################################
#     Load Data From VBA API     #
##################################
def load_data():
	perPage = 50

	it = 0
	data = []
	while True:
		it += 1
		print('#{}: {}'.format(it, len(data)))
		url = 'https://api.vbagame.com/api/v1/team/all?page={}&perPage={}'.format(it, perPage)
		r = requests.get(url)
		j = r.json()
		if len(j['teams']) == 0:
			break
		data += j['teams']
	teams = pd.DataFrame(data)
	teams[[ 'teamName','id' ]].to_csv(base_dir+'vba_teams.csv', index=False)
	teams[teams.teamName == 'Tuck Frump']

	teams.head()
	ids = sorted(teams.id.unique())
	tot = len(ids)
	it = 0
	games_data = []
	for team_id in ids:
		it += 1
		print('#{} / {} ({})'.format(it, tot, len(games_data)))
		page = 1
		while True:
			url = 'https://api.vbagame.com/api/v1/team/{}/games?page={}&perPage={}'.format(team_id, page, perPage)
			r = requests.get(url)
			j = r.json()
			games_data += j['data']['schedule']
			if j['meta']['total'] < perPage:
				break
	game_df = pd.DataFrame(games_data)
	game_df['homeId'] = game_df.homeUser.apply(lambda x: x['id'] if x and 'id' in x.keys() else None )
	game_df['awayId'] = game_df.awayUser.apply(lambda x: x['id'] if x and 'id' in x.keys() else None )

	game_df[[ 'homeId','awayId','homeScore','awayScore','matchResults','matchAction','gameStartTime' ]].dropna().to_csv(base_dir+'vba_games.csv', index=False)

###################################
#     Run Power Ratings Model     #
###################################
def run_model():
	teams = pd.read_csv(base_dir+'vba_teams.csv')
	game_df = pd.read_csv(base_dir+'vba_games.csv')
	game_df = game_df[ (game_df.homeScore > 0) & (game_df.awayScore > 0) ].drop_duplicates()
	game_df['homeId'] = game_df.homeId.astype(str)
	game_df['awayId'] = game_df.awayId.astype(str)

	game_df[(game_df.homeId == '678') | (game_df.awayId == '678')][[ 'homeId','awayId','homeScore','awayScore','matchResults','matchAction' ]].drop_duplicates()

	h = game_df[game_df.matchResults.isin(['LOSS','WIN'])][[ 'homeId','awayId','homeScore','awayScore' ]]
	a = game_df[game_df.matchResults.isin(['LOSS','WIN'])][[ 'awayId','homeId','awayScore','homeScore' ]]
	h.columns = [ 't1','t2','s1','s2' ]
	a.columns = [ 't1','t2','s1','s2' ]
	h['is_home'] = 1
	a['is_home'] = 0
	train = h.append(a).drop_duplicates()

	t1 = pd.get_dummies(train.t1)
	t1.columns = ['t1_'+c for c in t1.columns]
	t2 = pd.get_dummies(train.t2)
	t2.columns = ['t2_'+c for c in t2.columns]
	train = pd.concat([train.reset_index(drop=True), t1.reset_index(drop=True), t2.reset_index(drop=True)], 1)
	train['score_dff'] = train.s1 - train.s2
	train['is_win'] = (train.s1 > train.s2).astype(int)
	pred_cols = list(t1.columns) + list(t2.columns)
	clf = RidgeCV(alphas=[0.01, .1, .3, 1, 3, 10])
	clf.fit(train[pred_cols].values, train.score_dff.values)
	train['pred_dff'] = clf.predict(train[pred_cols].values)
	clf_log = LogisticRegression()
	clf_log.fit(train[['pred_dff']].values, train.is_win.values)
	clf_log.coef_
	train['pred_prob'] = clf_log.predict_proba(train[['pred_dff']].values)[:,1]
	train.groupby('is_win').pred_prob.mean()
	train.sort_values('pred_prob').head(20)[['t1','t2','pred_prob','score_dff']]
	train.sort_values('pred_prob').tail(20)[['t1','t2','pred_prob','score_dff']]

	test_1 = train[['t1']].drop_duplicates()
	test_2 = train[['t2']].drop_duplicates()
	test_1['m'] = 1
	test_2['m'] = 1
	test = test_1.merge(test_2)
	t1 = pd.get_dummies(test.t1)
	t2 = pd.get_dummies(test.t2)
	t1.columns = ['t1_'+c for c in t1.columns]
	t2.columns = ['t2_'+c for c in t2.columns]

	test = pd.concat([test.reset_index(drop=True), t1.reset_index(drop=True), t2.reset_index(drop=True)], 1)
	test['pred_dff'] = clf.predict(test[pred_cols].values)
	test['pred_prob'] = clf_log.predict_proba(test[['pred_dff']].values)[:,1]

	test[ (test.t1 == "Killer 3's") & (test.t2 == "Bivouac") ]
	test[ (test.t2 == "Killer 3's") & (test.t1 == "Bivouac") ]
	test.t2.unique()

	train[train.t1 == 'saga']

	n_games = train.groupby('t1').t2.count().reset_index().rename(columns={'t2':'n_games'})
	train['is_loss_by_u5'] = ((train.score_dff < 0) & (train.score_dff >= -5)).astype(int)
	g = train.groupby('t1')[['is_win','score_dff','is_loss_by_u5']].mean().reset_index().rename(columns={'is_win':'win_pct'})
	rk = test.groupby('t1').pred_prob.mean().reset_index().sort_values('pred_prob', ascending=0)
	rk['power_rating'] = range(len(rk))
	rk['power_rating'] = rk['power_rating'] + 1
	rk = rk.merge(teams.rename(columns={'id':'t1'}))
	rk = rk.merge(n_games)
	rk = rk.merge(g)
	sch = train[['t1','t2']].merge(rk[['t1','pred_prob']]).groupby('t2').pred_prob.mean().reset_index()
	sch.columns = [ 't1','schedule_difficulty' ]
	rk = rk.merge(sch)
	rk['dff'] = rk.pred_prob - rk.win_pct
	rk = rk[[ 'power_rating','teamName','pred_prob','n_games','win_pct','score_dff','schedule_difficulty','dff','is_loss_by_u5' ]]
	rk.columns = [ 'Power Rating','Team','Modeled Win %','Games','Win %','Avg Score Dff','Schedule Difficulty','Modeled vs Actual','% Games <= 5pt Loss' ]
	rk.to_csv(base_dir+'vba_power_ratings.csv', index=False)

