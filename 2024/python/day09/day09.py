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
# ##
# from collections import defaultdict
# from dataclasses import dataclass
# from typing import Optional
#
# import numpy as np
#
#
# @dataclass
# class Chunk:
#     size: int
#     pos: int = -1
#     prev: Optional["Empty"] = None
#     next: Optional["Empty"] = None
#
#     def insert_after(self, chunk):
#         chunk.next = self.next
#         chunk.prev = self
#         chunk.pos = self.pos + 1
#         if self.next:
#             self.next.prev = chunk
#             self.next.pos = chunk.pos + 1
#         self.next = chunk
#         return chunk
#
#     def detach(self):
#         e = Empty(size=self.size, pos=0)
#         if self.prev:
#             self.prev.next = e
#             e.prev = self.prev
#             e.pos = self.prev.pos + 1
#         if self.next:
#             self.next.prev = e
#             e.next = self.next
#         self.prev = None
#         self.next = None
#         self.pos = -1
#         return self
#
#     def remove(self):
#         if self.prev:
#             self.prev.next = self.next
#         if self.next:
#             self.next.prev = self.prev
#
#     def print_chain(self):
#         n = self
#         while n is not None:
#             print(f"{n} ", end="")
#             n = n.next
#         print()
#
#     def __iter__(self):
#         n = self
#         while n:
#             yield n
#             n = n.next
#
#
# @dataclass
# class File(Chunk):
#     id: int = -1
#
#     def __repr__(self):
#         return f"({self.pos}, {self.id}, {self.size})"
#
#
# @dataclass
# class Empty(Chunk):
#     def __repr__(self):
#         return f"({self.pos}, _, {self.size})"
#
#
# def print_map(chunks):
#     for chunk in chunks:
#         if isinstance(chunk, File):
#             print(f"{chunk.id}" * chunk.size, end="")
#         elif isinstance(chunk, Empty):
#             print("." * chunk.size, end="")
#     print()
#
#
# disk_map = [int(x) for x in open("day9.txt").read().strip()]
# files, empties = disk_map[::2], disk_map[1::2]
# files = [(i, size) for i, size in enumerate(files)]
# chunks = last_chunk = File(id=files[0][0], size=files[0][1], pos=0)
# for empty_size, (file_id, file_size) in zip(empties, files[1:]):
#     last_chunk = last_chunk.insert_after(Empty(size=empty_size))
#     last_chunk = last_chunk.insert_after(File(id=file_id, size=file_size))
#
# files_by_size = defaultdict(list)
# for chunk in iter(chunks):
#     if isinstance(chunk, File):
#         files_by_size[chunk.size].insert(0, chunk)
#
# # print_map(chunks)
# # chunks.print_chain()
# # print(files_by_size)
#
# empty = chunks
# to_place_next = last_chunk
# while to_place_next:
#     # Grab the last file.
#     done = False
#     while not isinstance(to_place_next, File):
#         if to_place_next is None:
#             done = True
#             break
#         to_place_next = to_place_next.prev
#     if done:
#         break
#
#     file, to_place_next = to_place_next, to_place_next.prev
#     print(file)
#
#     # Place the file in the next empty chunk.
#     found = False
#     empty = chunks
#     while not found:
#         if isinstance(empty, Empty) and empty.size >= file.size:
#             found = True
#             break
#         else:
#             empty = empty.next
#             if empty is None or empty.pos >= file.pos:
#                 break
#     if not found:
#         # print("No place found. Moving on to", to_place_next)
#         # to_place_next = to_place_next.prev
#         continue
#
#     # chunks.print_chain()
#     # print("Placing", file, "in", empty)
#
#     if file.size == empty.size:
#         # Perfect fit. Discard the Empty.
#         empty.prev.insert_after(file.detach())
#         empty.remove()
#         empty = file.next
#     elif file.size < empty.size:
#         # Shrink available space in the Empty.
#         empty.prev.insert_after(file.detach())
#         empty.size -= file.size
#     elif file.size > empty.size:
#         # Break up the file. Discard the Empty.
#         to_place_next = file.prev.insert_after(
#             File(id=file.id, size=file.size - empty.size)
#         )
#         file.size = empty.size
#         empty.prev.insert_after(file.detach())
#         empty.remove()
#         empty = file.next
#     # print_map(chunks)
#     for c in iter(chunks):
#         if c.next:
#             assert c.next.prev == c
#         if c.prev:
#             assert c.prev.next == c
#
# # chunks.print_chain()
# # print_map(chunks)
#
# checksum = 0
# offset = 0
# for chunk in iter(chunks):
#     if not isinstance(chunk, File):
#         offset += chunk.size
#         continue
#     checksum += chunk.id * (sum(range(chunk.size)) + chunk.size * offset)
#     offset += chunk.size
# print("part 2": checksum)
