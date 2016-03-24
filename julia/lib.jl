function loaddata(fname, resolution, n)
    fp = open(fname)
    return Mmap.mmap(fp, Matrix{UInt64}, (resolution, n))
end

function loadmap(fname)
    fp = open(fname, "r")
    bytes = readbytes(fp)
    close(fp)
    string = UTF8String(bytes)
    lines = split(string, "\n")
    table = Dict{UTF8String, Int64}([s => i for (i, s) in enumerate(lines)])
    return (lines, table)
end

function norm_id(x)
    return 1.0
end

function norm_max(x)
    return 1.0 / maximum(x)
end

function norm_avg(x)
    avg = sum(x) / length(x)
    return 1.0 / avg
end

function transform_id(x)
    return x
end

function transform_log(x)
    return log(1.0 + x)
end

function distance_simple(data, i, fnorm=norm_id, ftransform=transform_id)
    n = size(data)[2]

    x_raw = ftransform(convert(Array{Float64}, data[:,i]))
    x_norm = fnorm(x_raw)
    x = x_raw * x_norm

    result = Array{Float64}(n)
    for j in 1:n
        y_raw = ftransform(convert(Array{Float64}, data[:,j]))
        y_norm = fnorm(y_raw)
        y = y_raw * y_norm

        delta = sum((x - y) .^ 2)
        result[j] = delta
    end
    return result
end
