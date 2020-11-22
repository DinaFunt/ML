import pandas as pd
import numpy as np

class Node():
    def __init__(self, to, s, r, a):
        self.to = to
        self.s = s
        self.r = r
        self.a = a

def affinity_propagation(max_iterarions):
    c = np.zeros(v_number, dtype=int)

    for _ in range(max_iterarions):

        for i in range(v_number):
            m = len(graph[i])
            max1 = -1000000000
            max2 = -1000000000
            argmax1 = -1
            for j in range(m):
                value = graph[i][j].a + graph[i][j].s
                if (value > max1):
                    max1, value = value, max1
                    argmax1 = j
                if (value > max2):
                    max2 = value
            # update responsibilities
            for k in range(m):
                if (k != argmax1):
                    graph[i][k].r = graph[i][k].s - max1
                else:
                    graph[i][k].r = graph[i][k].s - max2



        for k in range(v_number):
            m = len(graph[k])

            sum = 0
            for i in range(m - 1):
                sum += max(0, graph[k][i].r)
            # update availabilities
            rkk = graph[k][m - 1].r
            for i in range(m - 1):
                graph[k][i].a = min(0, rkk + sum - max(0, graph[k][i].r))
            graph[k][m - 1].a = sum

    # set claster labels
    for i in range(v_number):
        m = len(graph[i])
        maxValue = -1000000000
        argmax = i
        for k in range(m):
            value = graph[i][k].a + graph[i][k].r
            if (value > maxValue):
                maxValue = value
                argmax = graph[i][k].to
        c[i] = argmax

    return c

def is_contains(top10, user_locations):
    for place in top10:
        if (place in user_locations):
            return True
    return False

def get_percent(user_locations, all_locations):
    for place in user_locations:
        all_locations[place] -= 1
    top10 = np.argsort(-all_locations)[:10]

    res = 0
    for place in top10:
        if (place in user_locations): res += 1
    return res / 10


if __name__ == '__main__':
    edges = np.genfromtxt("Gowalla_edges.txt", dtype=int, delimiter='\t')

    v_number = np.amax(edges) + 1
    graph = [[] for v in range(v_number)]
    for v_from, v_to in edges:
        graph[v_from].append(Node(v_to, 1, 0, 0))
    for k in range(v_number):
        graph[k].append(Node(k, -1.5, 0, 0))

    # get clusters
    c = affinity_propagation(max_iterarions=5)

    totalCheckins = pd.read_csv('Gowalla_totalCheckins.txt', sep="\t", header=None, names=['user', 'check-in-time', 'latitude', 'longitude', 'location-id'])
    users_with_locations = totalCheckins.groupby('user')['location-id'].apply(list)
    l = len(users_with_locations)


    percents = 0.0
    m = max(totalCheckins['location-id']) + 1
    user_number = l * 0.05

    for cluster in np.unique(c):
        if (user_number < 0): break
        locations = np.zeros(m, dtype=int)

        # get users of current cluster
        users_of_cluster = np.nonzero(c == cluster)[0]

        # count how many times users from the cluster of a specific hidden user have registered in a specific place
        for user in users_of_cluster:
            if (user in users_with_locations):
                locations[users_with_locations[user]] += 1

        # get top 10 checked in places
        top10 = np.argsort(-locations)[:10]

        # count the percentage
        for user in users_of_cluster:
            if (user_number < 0): break
            user_number -= 1
            if (user in users_with_locations):
                for place in top10:
                    if (place in users_with_locations[user]):
                        percents += get_percent(users_with_locations[user], np.copy(locations))
                        break

    print(percents / (l * 0.05) * 100)
