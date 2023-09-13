
# Import the modules that we need for the project
import neat
import os
from Stats import Stats
import random


"""
randdata picks a random dataset from our list of datasets in order to train the NEAT AI properly
After it picks the random data. It will read the data.
"""

def randdata():
    lstOfData = [
        "2018-2023",
        "2018Data",
        "2020Data",
        "2021Data",
        "2022AllSpyG",
        "2022MarSpyG",
        "ALLDATA",
        "MaxDataSpyG",
        "Recentdata",
        "Sept-March2023"
    ]
    num = random.randint(0, len(lstOfData) - 1)
    choice = lstOfData[num]
    path = "Datas/" + choice
    data2022 = open(path)
    lines = data2022.readlines()
    # Data is put in lines var until we use a for loop to cut off the /n at the end of each line and add it to data var
    data = []
    for line in lines:
        num = float(line[0:len(line) - 2])
        data.append(num)
    return data, path

# Amount of money each child starts with (Add user input using following line if desired)
# eval(input("How much money would you like the AI's to start with?"))
startMoney = 500


# defining Main with genomes and config links it into the NEAT alg with the genomes being the learned data
def main(genomes, config):
    # Neat needs nets and ge. nets determines the network. ge is the specific genomes
    nets = []
    ge = []
    players = []
    """
    Since we don't want to mess with the player order in genomes, we need to place _ as a placeholder.
    g is used to grade fitness and also set fitness to 0 at the start of each round.
    """
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        # money / num of stocks
        players.append([0, 0, []])
        g.fitness = 0
        ge.append(g)

    # set player inventories
    # NOTE PLAYER[0] IS MONEY; PLAYER [1] IS THE NUMBER OF STOCKS HELD

    for player in players:
        stocksHeld = 0
        currentMoney = startMoney
        player[0] = currentMoney
        player[1] = stocksHeld

    # Create some normal vars. Day number and Day price for the day we are on and the price of the stock on that day.
    dayNum = 0
    dayPrice = 0

    #Grab dataset using the randdata function
    data, path = randdata()

    # Start loop day = the price for that day
    for day in data:
        dayPrice = day
        dayNum += 1

        # sight defines how many days in the past can be viewed by the AI (20-30 is what I've found is best)
        sight = 30

        """
        These next lines will set the sight to the max number possible if there isn't enough datapoints to reach 
        sight value
        """
        if sight > dayNum:
            sight = dayNum + 1

        # Find the mean and SD of the past few days so the AI can have some information to make decisions
        """
        1: Make view var to contain all the data within the viewable range (sight var)
        2: Go through each of the past x days (x = sight) and add those values to the view var
        3: Make view compatible with Stats module and obtain stats from those.
        """
        view = []
        for i in range(-1 * sight, 0):
            yesterday = dayNum + i
            view.append(data[yesterday])
        statView = Stats(view)
        viewMean = Stats.getMean(statView)
        viewSD = Stats.getSD(statView)
        zScore = (dayPrice - viewMean) / viewSD
        # If sight is 0 SD and ZScore Really don't matter so I got rid of them
        if sight < 10:
            zScore = 0


        # This playerNum for loop goes through each player in the game
        for playerNum in range(len(players)):
            player = players[playerNum]

            # Next we will need to give the AI data and take a response from him (used while True to validate data)
            while True:
                """
                1: output var will hold the response of the AI
                2: I'm activating the neural network for the player playing by sending it info that we got from Stats
                3: output var comes out as a list of responses in decision var we pick the highest valued decision by AI
                4: We also interpreted that decision using the location that held the highest value
                Output meanings:
                output[0] = desire to buy
                output[1] = desire to sell
                output[2] = desire to hold
                output[3] = # of stocks to buy
                """
                output = nets[playerNum].activate([dayNum, dayPrice, viewMean, zScore, player[0], player[1]])
                decision = output.index(max(output[:3]))
                if decision == 0:
                    doDecision = "Buy"
                elif decision == 1:
                    doDecision = "Sell"
                else:
                    doDecision = "Hold"

                # Used howMany var to capture number of stocks to buy then calc how much it cost
                howMany = output[3]
                moneySpent = howMany * dayPrice

                # Before buying, check AI's will and the amount of money the AI has
                if doDecision == "Buy" and moneySpent <= player[0]:
                    # Do a buy (subtract money. Add stocks.)
                    player[0] -= moneySpent
                    player[1] += howMany

                    """
                    It is possible if the AI is dumb that it will say yes to buy but 0 to how many to buy.
                    To discourage this we are adding the following if statement
                    """
                    if howMany == 0 or moneySpent < 10:
                      ge[playerNum].fitness -= 1
                    else:
                      ge[playerNum].fitness += 1
                      player[2].append("Player %d bought %f for %f each ($%f) on day %d (%dDay z: %f  M: %f)"
                                       % (playerNum, howMany, dayPrice, moneySpent, dayNum, sight, zScore, viewMean))

                    # Now that the AI had decided. Use break to escape validation loop.
                    break

                # Before selling, make sure of AI intent and the stocks in its inventory
                elif doDecision == "Sell" and player[1] > howMany:
                    # Do a sell (Add money. subtract stocks.)
                    moneyReceived = howMany * dayPrice
                    player[0] += moneyReceived
                    player[1] -= howMany

                    # Again with the idea that the AI may try to sell and sell 0 stocks. We are punishing them.
                    if howMany == 0 or moneyReceived < 10:
                        ge[playerNum].fitness -= 1

                    # Reward them for successful behavior and notify user
                    else:
                        ge[playerNum].fitness += 1
                        player[2].append("Player %d sold %f for %f each ($%f) on day %d (%dDay z: %f  M: %f)"
                                         %
                                         (playerNum, howMany, dayPrice, moneyReceived, dayNum, sight, zScore, viewMean))
                    # Now that the AI had decided. Use break to escape validation loop.
                    break

                # If AI holds, do nothing
                elif doDecision == "Hold":
                    #print("Player", playerNum, "held on day", dayNum)
                    # Now that the AI had decided. Use break to escape validation loop.
                    break

                # For speed, we are also going to break the loop of the AI makes a mistake, but we will punish them.
                else:
                    ge[playerNum].fitness -= 1
                    break

    # Score players
    winnerScore = 0
    playerWinner = 0
    winnerFitness = 0
    winnerGainPercent = 0

    # We are making a for loop to go through each player and rank them by how well they did. Saved score in fitLevel
    # Player[0] is money; player[1] is stocks held
    for playerNum in range(0, len(players)):
        player = players[playerNum]
        liquidAssets = (player[1] * dayPrice) + player[0]
        gainPercent = (liquidAssets - startMoney) / startMoney
        gameLength = dayNum / 251
        fitLevel = (gainPercent / gameLength) * 100
        ge[playerNum].fitness = fitLevel
        # Punish players who did nothing, but reward players that did something to help AI at the start.
        if gainPercent <= 0:
            ge[playerNum].fitness = -10

        else:
            ge[playerNum].fitness += 5

        newFitLevel = ge[playerNum].fitness
        # if a player has the highest score, save their player number and fitness level
        if liquidAssets > winnerScore:
            winnerScore = liquidAssets
            playerWinner = playerNum
            winnerFitness = newFitLevel
            winnerGainPercent = fitLevel


    # Display the winning score/player
    print("*** Winner score: ", winnerScore, "*** on path:", path)
    print("""
    Winner num: %d
    Winner fit: %f
    Winner gain percent: %f 
    """ % (playerWinner, winnerFitness, winnerGainPercent))

    # prints the info for the player winner then a blank line.
    for i in players[playerWinner][2]:
        print(i)

    print()



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-836')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(main, 1000)




if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feeforward.txt")
    run(config_path)