import re
from subprocess import check_output, CalledProcessError
from pathlib import Path
from time import time

files = sorted(Path(".").glob("day*/day*.py"))
for file in files:
    day = int(re.match(r"day(\d+)", str(file)).group(1))
    try:
        start_time = time()
        output = check_output(["python", file.name], cwd=file.parent)
        end_time = time()
        part1 = ""
        part2 = ""
        for line in output.decode("utf-8").split("\n"):
            if m := re.match(r"^part 1: (.*)$", line):
                part1 = m.group(1)
            if m := re.match(r"^part 2: (.*)$", line):
                part2 = m.group(1)
    except CalledProcessError:
        part1 = "error"
        part2 = "error"
    print(f"day {day:02d}    part1: {part1.strip():<20} part2: {part2.strip():<40} time: {end_time - start_time:.4f}s")
