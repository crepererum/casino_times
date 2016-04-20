using DataFrames
using Gadfly

include("../DynamicTimeWarp.jl/src/DynamicTimeWarp.jl")

function gengausskernel(size, sigma)
    kernel = [
        1.0 / (sigma * sqrt(2.0 * pi)) * exp(-(x * x) / (2 * sigma * sigma))
        for x in -size:size
    ]
    return kernel / sum(kernel)
end

kernel_gauss_sigma1 = gengausskernel(10, 1)
kernel_gauss_sigma2 = gengausskernel(10, 2)
kernel_gauss_sigma3 = gengausskernel(10, 3)

function applykernel(x, kernel)
    n = size(x)[1]
    m = (size(kernel)[1] - 1) >> 1
    result = Array{Float64}(n)
    for i in 1:n
        accu = 0.0
        w = 0.0
        for d in -m:m
            i2 = i + d
            if i2 >= 1 && i2 <= n
                k = kernel[d + m + 1]
                accu += k * x[i2]
                w += k
            end
        end
        result[i] = accu / w
    end
    return result
end

function loaddata(fname, resolution, n, t=UInt64)
    fp = open(fname)
    return Mmap.mmap(fp, Matrix{t}, (resolution, n))
end

function loaddata_wd(fname, resolution, n, t=Float64)
    fp = open(fname, "w+")
    return Mmap.mmap(fp, Matrix{t}, (resolution, n))
end

function loadmap(fname)
    fp = open(fname, "r")
    bytes = readbytes(fp)
    close(fp)
    string = UTF8String(bytes)
    lines = filter(s -> length(s) > 0, split(string, "\n"))
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

function transform_smooth1(x)
    return applykernel(x, kernel_gauss_sigma1)
end

function transform_smooth2(x)
    return applykernel(x, kernel_gauss_sigma2)
end

function transform_log_smooth1(x)
    return transform_smooth1(transform_log(x))
end

function transform_log_smooth2(x)
    return transform_smooth2(transform_log(x))
end

function transform_gradient(x)
    return gradient(x)
end

function transform_loggradient(x)
    return gradient(transform_log(x))
end

function transform_gradient_smooth1(x)
    return gradient(transform_smooth1(x))
end

function transform_gradient_smooth2(x)
    return gradient(transform_smooth2(x))
end

function transform_loggradient_smooth1(x)
    return gradient(transform_log_smooth1(x))
end

function transform_loggradient_smooth2(x)
    return gradient(transform_log_smooth2(x))
end

function get_from_data(data, i, fnorm=norm_id, ftransform=transform_id)
    x_transformed = ftransform(convert(Array{Float64}, data[:,i]))
    return x_transformed * fnorm(x_transformed)
end

function dist_quad(x, y)
    return sum((x - y) .^ 2)
end

function dist_dtw(x, y)
    costs, _, _ = DynamicTimeWarp.dtw(x, y)
    return costs
end

function find_min_dists(data, i, fnorm=norm_id, ftransform=transform_id, fdist=dist_quad)
    n = size(data)[2]

    x = get_from_data(data, i, fnorm, ftransform)

    results = Array{Tuple{Int64, Float64}}(n)
    for j in 1:n
        y = get_from_data(data, j, fnorm, ftransform)
        d = fdist(x, y)
        results[j] = (j, d)
    end

    return sort(results, by=x -> x[2])
end

function plot_dataframe(df, title, geo=Geom.line, th=Theme(default_point_size=0.75mm))
    min_x = minimum(df[:x])
    max_x = maximum(df[:x])

    min_y = minimum(df[:y])
    max_y = maximum(df[:y])
    range_y = max_y - min_y
    margin_y = range_y * 0.05

    Gadfly.plot(
        df,
        x="x",
        y="y",
        color="label",
        geo,
        Guide.xlabel("Year"),
        Guide.ylabel("n"),
        Guide.colorkey("ngram"),
        Guide.title(title),
        Coord.Cartesian(xmin=min_x, xmax=max_x, ymin=(min_y - margin_y), ymax=(max_y + margin_y)),
        th
    )
end

function plot_all_that(data, map_s2i, t, ngrams, fnorm, ftransform, title, geo=Geom.line, th=Theme(default_point_size=0.75mm))
    dfs = map(ng -> DataFrame(x=t, y=get_from_data(data, map_s2i[ng], fnorm, ftransform), label=ng), ngrams)
    df = reduce(vcat, DataFrame(), dfs)
    plot_dataframe(df, title, geo, th)
end

function map_to_frequency(data, map_s2i, ftransform, ng)
    x = get_from_data(data, map_s2i[ng], norm_id, ftransform)
    abs(rfft(x ./ sum(x))[2:end])
end

function print_frequency_report(data, map_s2i, m, ngrams, ftransform, title)
    xvals = collect(1:(m / 2))
    dfs = map(ng -> DataFrame(x=xvals, y=map_to_frequency(data, map_s2i, ftransform, ng), label=ng), ngrams)
    df = reduce(vcat, DataFrame(), dfs)
    Gadfly.plot(
        df,
        x="x",
        y="y",
        color="label",
        Geom.point,
        Guide.xlabel("frequency"),
        Guide.ylabel("amplitude"),
        Guide.colorkey("ngram"),
        Guide.title(@sprintf("Frequency report - %s", title)),
        Guide.xticks(ticks=collect(1:(m / 16):(m / 2))),
        Scale.y_log10,
        Theme(default_point_size=0.5mm)
    )
end
