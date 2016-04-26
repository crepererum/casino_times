#!/usr/bin/env julia

fmap = ARGS[1]
fd1  = ARGS[2]
fd2  = ARGS[3]

include("../julia/lib.jl")

map_i2s, map_s2i = loadmap(fmap)
n = length(map_i2s)

fp1 = open(fd1)
fp2 = open(fd2)

data1 = Mmap.mmap(fp1, Matrix{Float64}, (n, 1))
data2 = Mmap.mmap(fp2, Matrix{Float64}, (n, 1))

norm = sort(data1[:,1])[2]

delta = (data2 - data1) ./ norm

@printf "mean:   %.5f\n" mean(delta)
@printf "std:    %.5f\n" std(delta)
@printf "min:    %.5f\n" minimum(delta)
@printf "q0.02:  %.5f\n" quantile(delta[:,1], 0.02)
@printf "q0.09:  %.5f\n" quantile(delta[:,1], 0.09)
@printf "q0.25:  %.5f\n" quantile(delta[:,1], 0.25)
@printf "median: %.5f\n" median(delta)
@printf "q0.75:  %.5f\n" quantile(delta[:,1], 0.75)
@printf "q0.91:  %.5f\n" quantile(delta[:,1], 0.91)
@printf "q0.98:  %.5f\n" quantile(delta[:,1], 0.98)
@printf "max:    %.5f\n" maximum(delta)

close(fp1)
close(fp2)
