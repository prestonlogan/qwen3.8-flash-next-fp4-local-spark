#!/usr/bin/env bash
# kill fp4local if MemAvailable drops below 1.5 GiB; log top RSS when below 12 GiB
while docker ps -q -f name=fp4local | grep -q .; do
  a=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
  echo "$(date +%T) $a" >> logs/mem.log
  if [ "$a" -lt 12582912 ]; then { echo "== $(date +%T) avail=$a"; grep -E "^(Cached|Shmem|AnonPages|Mlocked|Unevictable|NvMap|Dirty)" /proc/meminfo | tr -s " " | tr "\n" ";"; echo; ps -eo rss,nlwp,comm --sort=-rss | head -8; } >> logs/memdetail.log; fi
  if [ "$a" -lt 1572864 ]; then echo "$(date) KILL MemAvailable=$a" >> logs/watchdog.log; docker kill fp4local; fi
  sleep 0.5
done
