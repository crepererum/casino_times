using Colors


function gradients2color(g1, g2)
    normalized1 = map(x -> max(-1.0, min(1.0, x)), g1)
    normalized2 = map(x -> max(-1.0, min(1.0, x)), g2)

    boosted1 = (abs(normalized1) .^ 0.3) .* sign(normalized1)
    boosted2 = (abs(normalized2) .^ 0.3) .* sign(normalized2)

    clab = map(
        xy -> Lab(
            50.0 * (abs(xy[1]) + abs(xy[2])),
            xy[1] * 100.0,
            xy[2] * 100.0
        ),
        zip(boosted1, boosted2)
    )
    crgb = map(c -> convert(RGB, c), clab)

    return convert(Array{RGB{Float64}}, crgb)
end


function gengausskernel(size, sigma)
    kernel = [
        1.0 / (sigma * sqrt(2.0 * pi)) * exp(-(x * x) / (2 * sigma * sigma))
        for x in -size:size
    ]
    return kernel / sum(kernel)
end


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


function idxlist2picture(data, idxlist, first, last)
    n = last - first + 1
    m = size(idxlist)[1]
    A = fill(
        RGB(0.0, 0.0, 0.0),
        (n, m)
    )
    kernel = gengausskernel(14, 10)

    for (i, j) in enumerate(idxlist)
        x_int = data[first:last,j]
        if sum(x_int) == 0
            continue
        end

        x_float = convert(Array{Float64}, x_int)
        x_smooth = applykernel(x_float, kernel)
        m = maximum(abs(x_float))
        g1 = gradient(x_float, 1) ./ m
        g2 = gradient(x_smooth, 1) ./ m
        c = gradients2color(g1, g2)

        A[:, i] = c
    end

    return convert(Image, A)
end
