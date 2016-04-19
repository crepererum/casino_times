#!/usr/bin/env julia

fmap = ARGS[1]
fin = ARGS[2]
fout = ARGS[3]
fn = ARGS[4]
fin_t = ARGS[5]

include("../julia/lib.jl")

t = Int64
if fin_t == "Float64"
    t = Float64
elseif fin_t == "Int64"
    t = Int64
else
    println("What datatype?")
    exit(1)
end

map_i2s, map_s2i = loadmap(fmap)
n = length(map_i2s)
m = 256
data_in = loaddata(fin, m, n, t)
data_out = loaddata_wd(fout, m, n, Float64)

fn_sym = parse(fn)

function transform_dispatch(x)
    eval(Expr(:call, fn_sym, x))
end

for i in 1:n
    data_out[:,i] = get_from_data(data_in, i, norm_id, transform_dispatch)
end
