import csv, numpy
from utils_functions import avg_genres_rank_by_ids, init_avg_ranks

updated_train = {} #train entries to change
User_Movie_rate = {} #train
users = {} #number pf *users* changed in train (should be under 100)
D_matrix = {} # output of D_test.csv
true_ranks = {} #according to test results
MovieGenres = {}
avg_rank_by_id = {}
avg_rank_counter = {}
predicted = {} # get dict of relevant csv 

with open("M_G.csv","r") as csvfile2:
    reader2 = csv.DictReader(csvfile2)
    # remember : field_names = ['Movie_ID', 'Genres']
    for row in reader2:
        Genres_list = row['Genres'].split(', ')
        MovieGenres[row['Movie_ID']]=Genres_list
csvfile2.close()

with open("U_M_test_results.csv","r") as csvfile:
    reader = csv.DictReader(csvfile)
    # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
    for row in reader:
        true_ranks[(row['User_ID'], row['Movie_ID'])] = float(row['rate'])
    csvfile.close()

with open("test_D.csv","r") as csvfile:
    reader = csv.DictReader(csvfile)
    # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
    for row in reader:
        D_matrix[(row['Movie_ID_1'], row['Movie_ID_2'])] = float(row['Similarity'])
    csvfile.close()

with open("U_M_train.csv","r") as csvfile:
    reader = csv.DictReader(csvfile)
    # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
    for row in reader:
        User_Movie_rate[(row['User_ID'],row['Movie_ID'])] = float(row['rate'])
csvfile.close()

with open("predicted.csv","r") as csvfile: 
    reader = csv.DictReader(csvfile)
    # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
    for row in reader:
        predicted[(row['User_ID'], row['Movie_ID'])] = float(row['Rank']) 
    csvfile.close()

# find the closest movies according to the D matrix
def get_closest_movies(movie, D_matrix):
    movies = []
    for key in D_matrix:
        if movie == key[0]:
            movies.append(key[1])
        elif movie == key[1]:
            movies.append(key[0])
    return movies

# find similar people to user1. they are similar if they have similar movies preferences.
def find_similar_people(user1, genre, genres_avg_by_id, similar_users):
    current_score = genres_avg_by_id[user1][genre]
    for key in genres_avg_by_id.keys():
        for subkey in genres_avg_by_id[key]:
            if subkey == genre and numpy.abs(genres_avg_by_id[key][subkey] - current_score) <= 0.01:
                similar_users.append(key)

# function that calls the similarity functions
def get_similars(user, movie):
    similars = []
    init_avg_ranks()
    genres_avg_by_id = avg_genres_rank_by_ids()
    for g in MovieGenres[movie]:
        find_similar_people(user, g, genres_avg_by_id, similars)

    return similars

# main
for (user,movie) in predicted.keys(): 
    if len(users.keys()) >= 100: #if there are more than 100 users to change in train
        break
    exp_res = predicted[(user, movie)] #expected results 
    true_res = true_ranks[(user,movie)] # true results
    if numpy.abs(exp_res - true_res) <= 2: #if the the predicted is close to the the true ones. (we want to ruin this)
        movies = get_closest_movies(movie, D_matrix)
        for i in movies:
            similar2i = get_similars(user, i)
            for u in similar2i:
                if len(users.keys()) >= 100:
                    break
                if User_Movie_rate.has_key((u,i)):
                    if not users.has_key(u):
                        users[u] = 1
                    if User_Movie_rate[(u, i)] < 2:
                        updated_train[(u,i)] = 5
                    elif User_Movie_rate[(u, i)] > 2:
                        updated_train[(u, i)] = 0.5
                    users[u] += 1

#output :)
with open('predicted.csv', 'w' ) as write_file: 
    writer = csv.writer(write_file, lineterminator='\n')
    fieldnames2 = ["User_ID" , "Movie_ID" ,"Rank"]
    writer.writerow(fieldnames2)
    for result in updated_train:
        writer.writerow([  result[0] , result[1] , updated_train[result]  ])
write_file.close
