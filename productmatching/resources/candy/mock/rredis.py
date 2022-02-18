import redis

r = redis.Redis()
queue_name = "uQueue-offerMatching-ng-offers-cz"

r.delete(queue_name)

for item in [1988088407, 1467159132, 2410803298,
             1841871778, 1920287373, 759828227,
             1444008578, 708376999, 3141413827,
             1658929862]:
    r.lpush(queue_name, item)
