import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

def game_data_csv():
  df = pd.read_csv('game_data.csv')

  dfTeam0 = df[df['team'] == df.team.drop_duplicates().iloc[0]]
  dfTeam1 = df[df['team'] == df.team.drop_duplicates().iloc[1]]
  dfBall = df[df['team'] == df.team.drop_duplicates().iloc[2]]

  listIdTeam0 = []
  listIdTeam1 = []
  listIdBall = []

  for i in range(len(dfTeam0.id.drop_duplicates())):
    listIdTeam0.append(int(dfTeam0.id.drop_duplicates().iloc[i]))

  for i in range(len(dfTeam1.id.drop_duplicates())):
    listIdTeam1.append(int(dfTeam1.id.drop_duplicates().iloc[i]))

  for i in range(len(dfBall.id.drop_duplicates())):
    listIdBall.append(int(dfBall.id.drop_duplicates().iloc[i]))

  listIdTotal = listIdTeam0 + listIdTeam1 + listIdBall

  listDfTeam0 = []
  listDfTeam1 = [] 
  listDfTeam0String = []
  listDfTeam1String = [] 

  for i in range(len(listIdTotal)):
    if(i in listIdTeam0):
      globals()['df%sTeam0' % i] = dfTeam0[dfTeam0['id'] == i]
      listDfTeam0String.append('df%sTeam0' % i)
      listDfTeam0.append(dfTeam0[dfTeam0['id'] == i])

  for i in range(len(listIdTotal)):
    if(i in listIdTeam1):
        globals()['df%sTeam1' % i] = dfTeam1[dfTeam1['id'] == i]
        listDfTeam1String.append('df%sTeam1' % i)
        listDfTeam1.append(dfTeam1[dfTeam1['id'] == i])

  def normalized_coordinates(dataset):
    listx = []
    listy = []
    for i in range(len(dataset)):
      x = ((dataset.iloc[i]['x_pos']) /  (480/9000)) - 4500
      y = ((dataset.iloc[i]['y_pos']) /  (340/6000)) - 3000
      x = -x
      listx.append(x)
      listy.append(y)      
    dataset.loc[:,'x_pos_game_data'] = listx
    dataset.loc[:,'y_pos_game_data'] = listy


  ###Create Csv for team0

  for i in range(len(listDfTeam0)):
      normalized_coordinates(listDfTeam0[i])

  ###Create Csv for team1
  for i in range(len(listDfTeam1)):
      normalized_coordinates(listDfTeam1[i])

  ###Create Csv for ball
  normalized_coordinates(dfBall)

  frames = []

  for i in range(len(listDfTeam0)):
    frames.append(listDfTeam0[i])

  for i in range(len(listDfTeam1)):
    frames.append(listDfTeam1[i])

  frames.append(dfBall)


  result_csv = pd.concat(frames)

  return result_csv
  #result_csv.to_csv('game_data_csv.csv', index=False)


def reverse_json_coordinates(playerList, df, is_reversed):
  for i in range(len(playerList)):
    listx = []
    listy = []
    dataset = playerList[i]
    for j in range(len(dataset)):
      pose = dataset.iloc[j]['pose']
      if(is_reversed):  
          x = float(pose[0])
          y = float(pose[1])
            #if (dataset.iloc[j].teamNum == df.teamNum.drop_duplicates().iloc[0]):
            #  x = -x
            #  y = -y
      else:
          x = -float(pose[0])
          y = -float(pose[1])
      if (dataset.iloc[j].teamNum == df.teamNum.drop_duplicates().iloc[0]):
          x = -x
          y = -y
      listx.append(x)
      listy.append(y)
    dataset.loc[:,'x_pos'] = listx
    dataset.loc[:,'y_pos'] = listy
  df_result_json = pd.concat(playerList)
  #df_result_json.to_csv('df_result_json.csv', index=False)
  return df_result_json

def json_to_csv(is_reversed):

  df = pd.read_json('teamcomm_2019-07-06_11-30-56-259_TJArk_SPQR Team_1stHalf.log.tc.json')

  dfTeam0 = df[df['teamNum'] == df.teamNum.drop_duplicates().iloc[0]]
  dfTeam1 = df[df['teamNum'] == df.teamNum.drop_duplicates().iloc[1]]

  playerListTeam0 = []
  playerListTeam1 = []

  for i in range(1, len(df.playerNum.drop_duplicates())+1, 1):
    playerListTeam0.append(dfTeam0[dfTeam0['playerNum'] == i])
    
  for i in range(1, len(df.playerNum.drop_duplicates())+1, 1):
    playerListTeam1.append(dfTeam1[dfTeam1['playerNum'] == i])

  playerList = playerListTeam0 + playerListTeam1

  return reverse_json_coordinates(playerList, df, is_reversed)

def orderPlayerDf(df, isCsv):
  listUpperField = []
  listDownField = []
  for i in range(len(df)):
    if(isCsv):
      if(df[i].iloc[0].y_pos_game_data > 0):
        listUpperField.append(df[i].iloc[0])
      else:
        listDownField.append(df[i].iloc[0])
    elif(df[i].iloc[0].y_pos > 0):
        listUpperField.append(df[i].iloc[0])
    else:
        listDownField.append(df[i].iloc[0])
  orderListUpperField = orderPlayer(listUpperField, 'ascending', isCsv)
  orderListDownField = orderPlayer(listDownField, 'descending', isCsv)
  return orderListUpperField + orderListDownField

def orderPlayer(listPlayer, order, isCsv):
  listPos = []
  listNum = []
  for i in range(len(listPlayer)):
    if(isCsv):
      listPos.append(listPlayer[i].x_pos_game_data)
      listNum.append(listPlayer[i].id)
    else:
      listPos.append(listPlayer[i].x_pos)
      listNum.append(listPlayer[i].playerNum)
  df = pd.DataFrame(list(zip(listPos, listNum)), columns = ['x_pos','num'])
  orderList = []
  if(order == 'ascending'):
    df = df.sort_values(by=['x_pos'], ascending=True)
  else:
    df = df.sort_values(by=['x_pos'], ascending = False)
  for i in range(len(df)):
    orderList.append(df.iloc[i].num)
  return(orderList)

def order_json():

  df = json_to_csv(False)

  df.loc[(df.teamNum == 19), 'teamNum'] = 0
  df.loc[(df.teamNum == 20), 'teamNum'] = 1

  dfTeam0 = df[df['teamNum'] == df.teamNum.drop_duplicates().iloc[0]]
  dfTeam1 = df[df['teamNum'] == df.teamNum.drop_duplicates().iloc[1]]

  if(dfTeam0.iloc[0].x_pos > 0):
    teamRight = dfTeam0
    teamLeft = dfTeam1
  else:
    teamRight = dfTeam1
    teamLeft = dfTeam0

  playerRight = []
  playerLeft = []

  for i in range(1, len(df.playerNum.drop_duplicates())+1, 1):
    globals()['dfplayer%sTeamRight' % i] = teamRight[teamRight['playerNum'] == i]
    playerRight.append(teamRight[teamRight['playerNum'] == i])

    
  for i in range(1, len(df.playerNum.drop_duplicates())+1, 1):
    globals()['dfplayer%sTeamLeft' % i] = teamLeft[teamLeft['playerNum'] == i]
    playerLeft.append(teamLeft[teamLeft['playerNum'] == i])

  rightListJson = orderPlayerDf(playerRight, False)
  leftListJson = orderPlayerDf(playerLeft, False)

  return rightListJson, leftListJson

def order_csv():

  df = game_data_csv()
  df = df[df.id != -1]

  dfTeam0 = df[df['team'] == df.team.drop_duplicates().iloc[0]]
  dfTeam1 = df[df['team'] == df.team.drop_duplicates().iloc[1]]

  listIdTeam0 = []
  listIdTeam1 = []

  for i in range(len(dfTeam0.id.drop_duplicates())):
    listIdTeam0.append(int(dfTeam0.id.drop_duplicates().iloc[i]))

  for i in range(len(dfTeam1.id.drop_duplicates())):
    listIdTeam1.append(int(dfTeam1.id.drop_duplicates().iloc[i]))

  listIdTotal = listIdTeam0 + listIdTeam1

  playerRight = []
  playerLeft = []

  for i in range(len(listIdTotal)):
    if(df[df.id == listIdTotal[i]].iloc[200].x_pos_game_data > 0):
      playerRight.append(df[df.id == listIdTotal[i]])
    else:
      playerLeft.append(df[df.id == listIdTotal[i]])

  rightListCsv = orderPlayerDf(playerRight, True)
  leftListCsv = orderPlayerDf(playerLeft, True)

  return df, leftListCsv, rightListCsv

def create_dataframe_order():
  df, leftListCsv, rightListCsv = order_csv()
  leftListJson, rightListJson = order_json()
  df['playerNum'] = ''
  for i in range(5):
    df.loc[df.id == leftListCsv[i], 'playerNum'] = leftListJson[i]
    df.loc[df.id == leftListCsv[i], 'team'] = 0
  for i in range(5):
    df.loc[df.id == rightListCsv[i], 'playerNum'] = rightListJson[i]
    df.loc[df.id == rightListCsv[i], 'team'] = 1
  df = df.drop(['id'], axis = 1)
  df.rename(columns={'playerNum': 'id'}, inplace=True) 
  return df


def calculatePointsJson(dataset, color, ax):
    for i in range(len(dataset)):
      x = dataset.iloc[i].x_pos
      y = dataset.iloc[i].y_pos
      centrePose = plt.Circle((x,y),30,color=color)
      #testo = plt.text(x, y, str((int(dataset.iloc[i].playerNum))), fontsize=50)
      ax.add_patch(centrePose)

def calculatePointsCsv(dataset, color, ax):
  for i in range(len(dataset)):
    x = dataset.iloc[i].x_pos_game_data
    y = dataset.iloc[i].y_pos_game_data 
    #centrePose = plt.Circle((x+x*0.1,y),30,color=color)  
    #testo = plt.text(x, y, str((int(dataset.iloc[i].id))), fontsize=50)
    centrePose = plt.Circle((x,y),30,color=color) 
    ax.add_patch(centrePose)


def plotField():

  df = json_to_csv(False)


  df.loc[(df.teamNum == 19), 'teamNum'] = 0
  df.loc[(df.teamNum == 20), 'teamNum'] = 1

  dfTeam0 = df[df['teamNum'] == df.teamNum.drop_duplicates().iloc[0]]
  dfTeam1 = df[df['teamNum'] == df.teamNum.drop_duplicates().iloc[1]]

  playerListTeam0 = []
  playerListTeam1 = []

  for i in range(1, len(df.playerNum.drop_duplicates())+1, 1):
      playerListTeam0.append(dfTeam0[dfTeam0['playerNum'] == i])
      
  for i in range(1, len(df.playerNum.drop_duplicates())+1, 1):
      playerListTeam1.append(dfTeam1[dfTeam1['playerNum'] == i])

  playerList = playerListTeam0 + playerListTeam1

  fig=plt.figure(figsize=(20,10))
  
  ax=fig.add_subplot(1,1,1)
  
  plt.plot([-4500,-4500],[-3000,3000], color="black")
  plt.plot([4500,4500],[-3000,3000], color="black")
  plt.plot([-4500,4500],[3000,3000], color="black")
  plt.plot([-4500,4500],[-3000,-3000], color="black")

  plt.plot([0,0],[-3000,3000], color="black")

  #Area grande Left
  plt.plot([-2850,-2850],[-2000,2000], color="black")
  plt.plot([-4500,-2850], [-2000,-2000], color="black")
  plt.plot([-4500,-2850], [2000,2000], color="black")
  #Area piccola Left
  plt.plot([-3900,-3900],[-1100,1100], color="black")
  plt.plot([-4500,-3900], [-1100,-1100], color="black")
  plt.plot([-4500,-3900], [1100,1100], color="black")

  #Area grande Right
  plt.plot([2850,2850],[2000,-2000], color="black")
  plt.plot([4500,2850], [2000,2000], color="black")
  plt.plot([4500,2850], [-2000,-2000], color="black")
  
  #Area piccola Right
  plt.plot([3900,3900],[1100,-1100], color="black")
  plt.plot([4500,3900], [1100,1100], color="black")
  plt.plot([4500,3900], [-1100,-1100], color="black")


  centreCircle = plt.Circle((0,0),1500,color="black",fill=False)
  centreSpot = plt.Circle((0,0),50,color="black")

  penaltyleft = plt.Circle((-3200, 0),50,color="black")
  penaltyright = plt.Circle((3200, 0),50,color="black")

  ax.add_patch(centreCircle)
  ax.add_patch(centreSpot)
  ax.add_patch(penaltyleft)
  ax.add_patch(penaltyright)


  for i in range(len(playerListTeam0)):
    calculatePointsJson(playerListTeam0[i].head(50), 'red', ax)
  
  for i in range(len(playerListTeam1)):
    calculatePointsJson(playerListTeam1[i].head(50), 'blue', ax)

  df = create_dataframe_order()

  dfTeam0 = df[df['team'] == df.team.drop_duplicates().iloc[0]]
  dfTeam1 = df[df['team'] == df.team.drop_duplicates().iloc[1]]

  listIdTeam0 = []
  listIdTeam1 = []

  for i in range(len(dfTeam0.id.drop_duplicates())):
    listIdTeam0.append(int(dfTeam0.id.drop_duplicates().iloc[i]))

  for i in range(len(dfTeam1.id.drop_duplicates())):
    listIdTeam1.append(int(dfTeam1.id.drop_duplicates().iloc[i]))


  listIdTotal = listIdTeam0 + listIdTeam1 

  listDfTeam0 = []
  listDfTeam1 = [] 
  listDfTeam0String = []
  listDfTeam1String = [] 

  for i in range(len(listIdTotal)):
    if(i in listIdTeam0):
      globals()['df%sTeam0' % i] = dfTeam0[dfTeam0['id'] == i]
      listDfTeam0String.append('df%sTeam0' % i)
      listDfTeam0.append(dfTeam0[dfTeam0['id'] == i])

  for i in range(len(listIdTotal)):
    if(i in listIdTeam1):
        globals()['df%sTeam1' % i] = dfTeam1[dfTeam1['id'] == i]
        listDfTeam1String.append('df%sTeam1' % i)
        listDfTeam1.append(dfTeam1[dfTeam1['id'] == i])


  for i in range(len(listDfTeam0)):
      calculatePointsCsv(listDfTeam0[i].head(400), 'yellow', ax)
      
  for i in range(len(listDfTeam1)):
      calculatePointsCsv(listDfTeam1[i].head(400), 'green', ax)

  plt.show()

plotField()