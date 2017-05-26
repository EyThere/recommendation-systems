import csv
import numpy as np
from EX2_A_305101669_205749674 import simple_model,calculate_Rtilda

avg_rank_by_id = {}
avg_rank_counter = {}
def init_avg_ranks(User_Movie_rate): #initialize the avg ranks before updating it according to the data
    for user, movie in User_Movie_rate:
        if not avg_rank_by_id.has_key(user):
            avg_rank_by_id[user] = {}
            avg_rank_counter[user] = {}


def avg_genres_rank_by_ids(User_Movie_rate, MovieGenres): #output is a dict of {user_id:{genre1:user1_avg_rank_for_genre1, genre2:user1_avg_rank_for_genre2}..}
    for user, movie in User_Movie_rate.keys():
        for g in MovieGenres[movie]:
            if avg_rank_by_id[user].has_key(g):
                avg_rank_by_id[user][g] += User_Movie_rate[(user, movie)]
            else:
                avg_rank_by_id[user][g] = User_Movie_rate[(user, movie)]
            if avg_rank_counter[user].has_key(g):
                avg_rank_counter[user][g] += 1
            else:
                avg_rank_counter[user][g] = 1
    for user in avg_rank_by_id.keys():
        for genre in avg_rank_by_id[user].keys():
            avg_rank_by_id[user][genre] = float(avg_rank_by_id[user][genre]) / float(avg_rank_counter[user][genre])
    return avg_rank_by_id

########################################################################################
########################################################################################
############################       results C    #####################################

if __name__ == '__main__':

    User_Movie_rate = {}
    MovieGenres = {}
    ########################################################################################
    ########################################################################################
    ################### read U_M_matrix into dictionaries ####################################
    with open("U_M_train.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
        for row in reader:
            User_Movie_rate[(row['User_ID'], row['Movie_ID'])] = float(row['rate'])
    csvfile.close()

    ########################################################################################
    ################### read Movies Genres into dictionaries ################################
    with open("M_G.csv", "r") as csvfile2:
        reader2 = csv.DictReader(csvfile2)
        # remember : field_names = ['Movie_ID', 'Genres']
        for row in reader2:
            Genres_list = row['Genres'].split(', ')
            MovieGenres[row['Movie_ID']] = Genres_list
    csvfile2.close()

    results_list = []
    test = {}

    with open("U_M_test.csv", "r") as read_file:
        # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
        reader3 = csv.DictReader(read_file)
        for row in reader3:
            test[(row['User_ID'], row['Movie_ID'])] = 0.0

    init_avg_ranks(User_Movie_rate)
    user2gener = avg_genres_rank_by_ids(User_Movie_rate, MovieGenres)
    test_Pred = simple_model(User_Movie_rate, test)
    pred, R_tilda, D_matrix, rank_results = calculate_Rtilda(User_Movie_rate, test)
    for row in test:
        lst = [user2gener[row[0]][g] for g in MovieGenres[row[1]] if g in user2gener[row[0]]]
        sum_others = test_Pred[row] + pred[row]
        if len(lst) > 0:
            sum_others = (sum_others + np.mean(lst))/3.0
        else:
            sum_others /= 2.0
        if sum_others < 0.5:
            sum_others = 0.5
        if sum_others > 5:
            sum_others = 5
        results_list.append( (row[0] ,row[1] , sum_others) )


    ########################################################################################
    ########################################################################################
    #######  output file ###################################################################
    with open('C.csv', 'w' ) as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["User_ID" , "Movie_ID" ,"Rank"]
        writer.writerow(fieldnames2)
        for result in results_list:
            writer.writerow([  result[0] , result[1] , result[2]  ])
    write_file.close




#######################################################################################
#######################################################################################


        