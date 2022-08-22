import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

team_gc = pd.read_json("data_team.json")
team_gc['x'] = (np.array(team_gc.pose.to_list())[:,0])
team_gc['y'] = (np.array(team_gc.pose.to_list())[:,1])
team_gc.to_csv("team_gc.csv")

i = 0
frame = 0


track_gc = pd.read_csv("game_data.csv")
track_gc.x_pos = (track_gc.x_pos * (9000/1020) - 4500)
track_gc.y_pos = (track_gc.y_pos * (6000/503) - 3000)

track_gc = track_gc.sort_values(by = "time")

fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(1,1,1)
plt.plot([-4500,-4500],[-3000,3000], color="black")
plt.plot([4500,4500],[-3000,3000], color="black")
plt.plot([-4500,4500],[3000,3000], color="black")
plt.plot([-4500,4500],[-3000,-3000], color="black")


while i < team_gc.shape[0]:
	df2 = (team_gc[i:i+10][['teamNum','playerNum','x','y']])
	i+=10
	frame += 30
	for index,row in df2.iterrows():
		print(row.teamNum)
		if(row.teamNum == 19 and row.playerNum == 3):
			centrePose = plt.Circle((int(row.x),int(row.y)),30,color="green")
			ax.add_patch(centrePose)
		if(row.teamNum == 20 and row.playerNum == 3):
			centrePose = plt.Circle((int(row.x),int(row.y)),30,color="red")
			ax.add_patch(centrePose)
plt.show()
