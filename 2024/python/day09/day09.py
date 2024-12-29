import numpy as np

disk_map = np.fromregex("day09.txt", r"(\d)", dtype=[("num", "int")])["num"].tolist()
chunks, empty = disk_map[::2], disk_map[1::2]
chunks = [(i, size) for i, size in enumerate(chunks)]

insert_point = 1
space_available = empty.pop(0)
to_place_id, to_place_size = chunks.pop(-1)
while insert_point < len(chunks):
    # print(files)
    # print(insert_point)
    space_to_take = min(space_available, to_place_size)
    chunks.insert(insert_point, (to_place_id, space_to_take))
    if to_place_size > 0:
        insert_point += 1
    to_place_size -= space_to_take
    if to_place_size == 0:
        to_place_id, to_place_size = chunks.pop(-1)
    space_available -= space_to_take
    if space_available == 0:
        space_available = empty.pop(0)
        insert_point += 1
chunks.append((to_place_id, to_place_size))

offsets = [0] + np.cumsum([chunk_size for _, chunk_size in chunks[:-1]]).tolist()

checksum = 0
for (chunk_id, chunk_size), offset in zip(chunks, offsets):
    checksum += chunk_id * (sum(range(chunk_size)) + chunk_size * offset)
print("part 1:", checksum)

##
def print_map(files, empty):
    chunks = sorted(files + empty)
    for chunk in chunks:
        if len(chunk) == 3:
            _, file_id, file_size = chunk
            print(f"{file_id}" * file_size, end="")
        else:
            _, empty_size = chunk
            print("." * empty_size, end="")
    print()

disk_map = [int(x) for x in open("day09.txt").read().strip()]
cumsum = [0]
for d in disk_map[:-1]:
    cumsum.append(cumsum[-1] + d)
chunks = list(zip(cumsum, disk_map))
files, empty = chunks[::2], chunks[1::2]
files = [(pos, i, size) for i, (pos, size) in enumerate(files)][::-1]
empty = [(pos, size) for pos, size in empty if size > 0]

for file_index, (file_pos, file_id, file_size) in enumerate(files):
    for empty_index, (empty_pos, empty_size) in enumerate(empty):
        if empty_pos >= file_pos:
            break
        if empty_size < file_size:
            continue

        if empty_size > file_size:
            empty[empty_index] = (empty_pos + file_size, empty_size - file_size)
        else:
            del empty[empty_index]
        files[file_index] = (empty_pos, file_id, file_size)
        empty.append((file_pos, file_size))
        break

    # # Consolidate empty blocks
    # empty = sorted(empty)
    # for i in range(len(empty)):
    #     pos1, size1 = empty[i]
    #     while i < len(empty) - 1:
    #         pos2, size2 = empty[i + 1]
    #         if pos2 == pos1 + size1:
    #             size1 += size2
    #             del empty[i + 1]
    #         else:
    #             break
    #     empty[i] = (pos1, size1)
    #     if i >= len(empty) - 1:
    #         break

    # print_map(files, empty)

files = sorted(files)
checksum = 0
for (file_pos, file_id, file_size) in files:
    checksum += file_id * (sum(range(file_size)) + file_size * file_pos)
print("part 2:", checksum)
