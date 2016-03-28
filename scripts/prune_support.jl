#!/usr/bin/env julia

fdata = ARGS[1]
fin = ARGS[2]
fout = ARGS[3]
threshold = parse(Int64, ARGS[4])

include("../julia/lib.jl")

map_i2s, map_s2i = loadmap(fin)
n = length(map_i2s)
m = 256
data = loaddata(fdata, m, n)

support = sum(data, 1)
good_ones = map(x -> x[1], filter(x -> x[2] >= threshold, enumerate(support)))

fp = open(fout, "w")
for x in good_ones
    write(fp, map_i2s[x])
    write(fp, "\n")
end
close(fp)
