from copy import deepcopy
from collections import defaultdict, Counter
import pandas as pd

AP_NUM = 1


def sum_dict(dict1: defaultdict, dict2: defaultdict) -> defaultdict:
    counter1 = Counter(dict1)
    counter2 = Counter(dict2)
    counter_sum = counter1 + counter2
    return dict(counter_sum)


def cluster(graph, avail_graph):
    nodes_num = len(graph)
    resp_graph = deepcopy(avail_graph)
    clusters = defaultdict(set)

    for _ in range(AP_NUM):
        print('Completed nodes: ', end='')
        for i in range(nodes_num):
            for k in range(nodes_num):
                sum = sum_dict(avail_graph[i], graph[i])
                if k in sum:
                    sum.pop(k)
                max_sum = max(sum.values()) if sum else 0

                cur_weight = 1 if k in graph[i] else 0
                resp_graph[i][k] = cur_weight - max_sum
            print(i, end=' ')
        print('\nThe first step of the AP-loop: done\n')

        print('Completed nodes: ', end='')
        for i in range(nodes_num):
            for k in range(nodes_num):
                resp_sum = 0
                for j in resp_graph[i]:
                    resp_sum += (resp_graph[j][k] if j not in {i, k} else max(0, resp_graph[j][k]))
                avail_graph[i][k] = min(0, resp_sum)
            avail_graph[i][i] = sum([max(0, resp_graph[j][i]) for j in resp_graph[i] if j != i])
            print(i, end=' ')
        print('\nThe second and third steps of the AP-loop: done\n')

    for i in range(nodes_num):
        max_sum = resp_graph[i][0] + avail_graph[i][0]
        cluster_num = 0
        for k in range(1, nodes_num):
            if resp_graph[i][k] + avail_graph[i][k] > max_sum:
                max_sum = resp_graph[i][k] + avail_graph[i][k]
                cluster_num = k
    clusters[cluster_num].add(i)
    del resp_graph, avail_graph
    return clusters


def check(clusters, ind):
    friends = clusters[ind].difference({ind})
    data = pd.read_csv('Gowalla_totalCheckins.txt', sep='\t', header=None)
    data.columns = ['user', 'check-in time', 'latitude', 'longitude', 'location id']

    friend_checkins = data.loc[data['user'].isin(friends)]
    checkins = defaultdict(int)
    for id in friend_checkins['location id']:
        checkins[id] += 1

    top_pred = sorted(checkins.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_pred = {entry[0] for entry in top_pred}
    gt_checkins = data.loc[data['user'] == ind]
    gt_checkins = set(gt_checkins['location id'])
    return 10 - len(top_pred.difference(gt_checkins))


def main():
    graph = defaultdict(lambda: defaultdict(int))
    avail_graph = defaultdict(lambda: defaultdict(int))
    with open('Gowalla_edges.txt', 'r') as file:
        for line in file.readlines():
            nodes = line.rstrip('\n').split('\t')
            n1 = int(nodes[0])
            n2 = int(nodes[1])
            graph[n1][n2] = 1
            avail_graph[n1][n2] = 0

    clusters = cluster(graph, avail_graph)
    print('The AP: done\n')

    miss_num = 0
    for i in clusters:
        miss_num += check(clusters, i)
    print('Misses number: ', miss_num)


if __name__ == '__main__':
    main()
