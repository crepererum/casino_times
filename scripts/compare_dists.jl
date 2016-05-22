#!/usr/bin/env julia

using StringDistances

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

function zero2one(x)
    if x == 0
        return 1
    else
        return x
    end
end

delta = (data2 - data1) ./ map(zero2one, data1)

@printf "mean:   %.5f\n" mean(delta)
@printf "std:    %.5f\n" std(delta)
@printf "min:    %.5f\n" minimum(delta)
@printf "q0.02:  %.5f\n" quantile(delta[:,1], 0.02)
@printf "q0.09:  %.5f\n" quantile(delta[:,1], 0.09)
@printf "q0.25:  %.5f\n" quantile(delta[:,1], 0.25)
@printf "q0.45:  %.5f\n" quantile(delta[:,1], 0.45)
@printf "median: %.5f\n" median(delta)
@printf "q0.55:  %.5f\n" quantile(delta[:,1], 0.55)
@printf "q0.75:  %.5f\n" quantile(delta[:,1], 0.75)
@printf "q0.91:  %.5f\n" quantile(delta[:,1], 0.91)
@printf "q0.98:  %.5f\n" quantile(delta[:,1], 0.98)
@printf "max:    %.5f\n" maximum(delta)


# dynamic limit calculation
function get_best_idx(sorted)
    data = collect(map(x -> x[2], sorted[2:convert(Int64, round(length(sorted) / 2))]))
    normalized = data - data[1]
    grad = gradient(normalized) ./ normalized
    grad2 = gradient(grad)
    indmax(grad2) # - 1 + 1
end
sorted1 = sort(collect(enumerate(data1[:, 1])), by=x -> x[2])
sorted2 = sort(collect(enumerate(data2[:, 1])), by=x -> x[2])
limit1 = get_best_idx(sorted1)
limit2 = get_best_idx(sorted2)
limit = limit1
@printf "limit1:  %i (<< choosen)\n" limit1
@printf "limit2:  %i\n" limit2

# string comparison
counter = 0
known = Dict{Int64, Int64}()
for entry in sorted1[1:limit]
    if !haskey(known, entry[1])
        known[entry[1]] = counter
        counter += 1
    end
end
newentries = 0
for entry in sorted2[1:limit]
    if !haskey(known, entry[1])
        known[entry[1]] = counter
        counter += 1
        newentries += 1
    end
end
string1 = join(map(x -> Char(known[x[1]]), sorted1[1:limit]))
string2 = join(map(x -> Char(known[x[1]]), sorted2[1:limit]))

@printf "sdiff:  %.5f\n" (1 - compare(Levenshtein(), string1, string2))
@printf "enew:   %.5f\n" (newentries / limit)


close(fp1)
close(fp2)
