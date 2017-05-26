import csv, numpy, math, random, time, copy



#get the r_avg value out of the data
def avg_rank(train):
    return numpy.mean(train.values())

#calculate bu/bi for each of the users/movies respectivly
def calculate_b_dicts(train, r_avg): #train is a dict of {(user_id, movie_id):rank}.
    bu_dict = {}
    bi_dict = {}
    bu_amount_dict = {}
    bi_amount_dict = {}
    for key in train:
        if bu_dict.has_key(key[0]):
            bu_dict[key[0]] += train[key]
            bu_amount_dict[key[0]] += 1
        else:
            bu_dict[key[0]] = train[key]
            bu_amount_dict[key[0]] = 1

        if bi_dict.has_key(key[1]):
            bi_dict[key[1]] += train[key]
            bi_amount_dict[key[1]] += 1
        else:
            bi_dict[key[1]] = train[key]
            bi_amount_dict[key[1]] = 1

    for key in bu_dict.keys():
        bu_dict[key] = ((bu_dict[key] / float(bu_amount_dict[key])) - r_avg)

    for key in bi_dict.keys():
        bi_dict[key] = ((bi_dict[key] / float(bi_amount_dict[key])) - r_avg)

    return bu_dict, bi_dict
    #todo: make the 5 to 5.0?

def simple_model(User_Movie_rate, test_data, toNormal = True):
    r_avg = avg_rank(User_Movie_rate)
    bu_dict, bi_dict = calculate_b_dicts(User_Movie_rate, r_avg)
    result_data = {}
    #update the bu/bu if there was no appearance in train data
    for pair in test_data.keys():
        if bu_dict.has_key(pair[0]):
            bu = bu_dict[pair[0]]
        else:
            bu = 0
        if bi_dict.has_key(pair[1]):
            bi = bi_dict[pair[1]]
        else:
            bi = 0
        result_data[pair] = r_avg + bu + bi
    # minimize mistake by limiting the predicted value to be in [0.5,5]
        if toNormal:
            if result_data[pair] > 5:
                result_data[pair] = 5
            if result_data[pair] < 0.5:
                result_data[pair] = 0.5
    return result_data

def distance_movies(movieId, movie, users):
    score = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for user in users:
        if (user, movieId) not in R_tilda: continue
        if (user, movie) not in R_tilda: continue
        score += R_tilda[(user, movie)] * R_tilda[(user, movieId)]
        norm1 += R_tilda[(user, movie)] ** 2
        norm2 += R_tilda[(user, movieId)] ** 2
    if norm1 == 0 or norm2 == 0:
        return 0
    score = score / (math.sqrt(norm1) * math.sqrt(norm2))
    return score

class MovieUserTuple:
    def __init__(self, id, rate):
        self.id = id;
        self.rate = rate;
    def __cmp__(self, other):
        c  = cmp(self.rate, other.rate)
        if c != 0: return c
        return -cmp(self.id, other.id)


def calculate_Rtilda(User_Movie_rate, test_data):
    train_movies = list(set([row[1] for row in User_Movie_rate.keys()]))
    train_users = list(set([row[0] for row in User_Movie_rate.keys()]))
    user2id = {user:j for j,user in enumerate(train_users)}
    RhatOnTrain = simple_model(User_Movie_rate, User_Movie_rate, False)
    RhatOnTest = simple_model(User_Movie_rate, test_data, False)
    R_tilda_mat = []
    sorted_m = list(sorted(train_movies, key=lambda v: int(v), reverse=True))
    movie2id_sorted = {movie: j for j, movie in enumerate(sorted_m)}
    movie2id = movie2id_sorted
    for user in train_users:
        R_tilda_mat.append([])
        for movie in sorted_m:
            if (user, movie) in User_Movie_rate:
                R_tilda_mat[-1].append(User_Movie_rate[(user, movie)] - RhatOnTrain[(user, movie)])
            else:
                R_tilda_mat[-1].append(0.0)
    R_tilda_mat = numpy.array(R_tilda_mat)
    norms_vector = numpy.linalg.norm(R_tilda_mat, axis = 0, keepdims=True).T
    D = (numpy.dot(R_tilda_mat.T, R_tilda_mat)) /  (numpy.dot(norms_vector ,norms_vector.T))
    predictedRates = {}
    D_dict = {}
    R_tilda = {}

    numpy.fill_diagonal(D, 0.0)
    S = numpy.argsort(-D, axis=1)[:,:4]

    for row in test_data.keys():
        movieId = row[1]
        userId = row[0]
        if movieId not in train_movies:
            predictedRates[row] = RhatOnTest[row]
            continue
        most_similar_movies = [sorted_m[idx] for idx in S[movie2id_sorted[movieId]]]
        for mv in most_similar_movies:
            D_dict[(movie2id[movieId], movie2id[mv])] = D[movie2id[movieId], movie2id[mv]]
        final_prediction = 0.0
        norm = 0.0
        for similar_movie in most_similar_movies:
            sim = D[movie2id[movieId], movie2id[similar_movie]]
            final_prediction += R_tilda_mat[user2id[userId], movie2id[similar_movie]] * sim
            norm += abs(sim)
        final_prediction /= norm
        final_prediction += RhatOnTest[row]
        if final_prediction < 0.5:
            final_prediction = 0.5
        if final_prediction > 5.0:
            final_prediction = 5.0
        predictedRates[row] = final_prediction
    for (user, movie) in User_Movie_rate:
        R_tilda[(user, movie)] = User_Movie_rate[(user, movie)] - RhatOnTrain[(user, movie)]
    return predictedRates, R_tilda, D_dict, [(k[0], k[1], predictedRates[k]) for k in predictedRates]

def loss(real, prediction):
    loss = 0.0
    for row in real.keys():
        loss += (real[row] - prediction[row])**2
    return math.sqrt(loss/len(real.keys()))


if __name__ == '__main__':

    User_Movie_rate = {}
    # {(U_i,M_j):rate , (U_m,M_n):rate , .....}
    User_Movie_rate_Part1 = {}
    User_Movie_rate_Part1_Pred = {}

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
    ########################################################################################
    ################### read U_M_Part1 into dictionaries ####################################
    with open("U_M_Part1.csv", "r") as csvfile3:
        reader = csv.DictReader(csvfile3)
        # remember : field_names = ['User_ID', 'Movie_ID','rate' ]
        for row in reader:
            User_Movie_rate_Part1[(row['User_ID'], row['Movie_ID'])] = float(row['rate'])
    csvfile3.close()

    ########################################################################################
    ########################################################################################
    ############################       results A  - 1     ######################################
    R_hat = []
    User_Movie_rate_Part1_Pred = simple_model(User_Movie_rate, User_Movie_rate_Part1)
    for row in User_Movie_rate_Part1_Pred.keys():
        R_hat.append( (row[0] ,row[1] , User_Movie_rate_Part1_Pred[row]  )  )

    #
    # ########################################################################################
    # ########################################################################################
    # ############################       results A - 2       ######################################

    pred, R_tilda, D_matrix, rank_results = calculate_Rtilda(User_Movie_rate, User_Movie_rate_Part1)

     ########################################################################################
     ########################################################################################
     ############################       results A - 3       ######################################

    RMSE = {}
    sum_basic = 0
    c_basic = len(User_Movie_rate_Part1)
    for r_ui in User_Movie_rate_Part1:
        diff = User_Movie_rate_Part1[r_ui] - User_Movie_rate_Part1_Pred[r_ui]
        pow_diff = math.pow(diff,2)
        sum_basic = sum_basic + pow_diff
    sum_basic = (sum_basic/c_basic)
    RMSE['Basic'] = math.sqrt(sum_basic)
    RMSE['Neighbours'] = loss(User_Movie_rate_Part1, pred)


    ########################################################################################
    ########################################################################################
    #######  output files ###################################################################
    with open('A1.csv', 'w' ) as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["User_ID" , "Movie_ID" ,"Rank"]
        writer.writerow(fieldnames2)
        for result in R_hat:
            writer.writerow([  result[0] , result[1] , result[2]  ])

    write_file.close


    with open('A2a.csv', 'w' ) as write_file:
         writer = csv.writer(write_file, lineterminator='\n')
         fieldnames2 = ["User_ID" , "Movie_ID" ,"difference"]
         writer.writerow(fieldnames2)
         for r in R_tilda:
             writer.writerow([  r[0] , r[1] , R_tilda[r]  ])
    write_file.close

    with open('A2b.csv', 'w' ) as write_file:
         writer = csv.writer(write_file, lineterminator='\n')
         fieldnames2 = ["Movie_ID_1" , "Movie_ID_2" ,"Similarity"]
         writer.writerow(fieldnames2)
         for d in D_matrix:
             writer.writerow([  d[0] , d[1] , D_matrix[d]  ])
    write_file.close


    with open('A2c.csv', 'w' ) as write_file:
         writer = csv.writer(write_file, lineterminator='\n')
         fieldnames2 = ["User_ID" , "Movie_ID" ,"Rank"]
         writer.writerow(fieldnames2)
         for result in rank_results:
             writer.writerow([  result[0] , result[1] , result[2]  ])
    write_file.close


    with open('RMSE.csv', 'w' ) as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["Method" , "RMSE"]
        writer.writerow(fieldnames2)
        for r in RMSE:
            writer.writerow([  r , RMSE[r]    ])
    write_file.close


    #######################################################################################
    #######################################################################################


        