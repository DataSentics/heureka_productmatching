import sys
import random
import redis

random.seed(0)

def main(
    path: str,
    queue: str,
    host: str,
    password: str
):
    print(f"Connecting to {host}.")

    client = redis.Redis(
        host=host,
        password=password
    )

    if input(f"Delete old items from {queue_name}?: ").strip().lower() in ("", "y", "yes"):
        print("Ok.., deleting.")
        client.delete(queue_name)

    try:
        size_to_push = int(input(f"Size to push from {path}: ").strip())
    except ValueError:
        size_to_push = 1

    if size_to_push > 0:
        print(f"Reading {path}.")
        with open(path, "r") as f, open("pushed.txt", "w") as pushed_f:
            lines = f.readlines()
            random.shuffle(lines)

            original_length = len(lines)
            lines = lines[:min(original_length, size_to_push)]

            print(f"Will push {len(lines)} out from {original_length}.")

            for index, line in enumerate(lines):
                if not line:
                    continue

                pushed_f.write(f"{line}\n")
                client.lpush(queue, line)

                if index % 100 == 0 or index + 1 == len(lines):
                    print(f"Pushed {index + 1} ids.")

    client.close()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        exit("Usage: rredis_monolith.py input_file input_queue host password")

    input_file, queue_name, host, password = sys.argv[1:]

    main(input_file, queue_name, host, password)
