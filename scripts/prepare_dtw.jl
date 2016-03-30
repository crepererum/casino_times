#!/usr/bin/env julia

fmap = ARGS[1]
fin = ARGS[2]
fout = ARGS[3]

include("../julia/lib.jl")

map_i2s, map_s2i = loadmap(fmap)
n = length(map_i2s)
m = 256
data_in = loaddata(fin, m, n)
data_out = loaddata_wd(fout, m, n)

for i in 1:n
    data_out[:,i] = get_from_data(data_in, i, norm_id, transform_loggradient)
end
