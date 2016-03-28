using PyPlot
using Wavelets

PyPlot.svg(false)

function plot_wavelet_report_helper(y, y_pruned, yt, yt_pruned, error, year_first, token, threshold)
    # metadata
    n = length(y)
    J = log2(n)
    x = collect(year_first:(year_first + n - 1))

    # prepare mask colormap
    cm = PyPlot.ColorMap("gray")
    cm[:_init]()
    colors = cm[:_lut]
    colors[1,4] = 0
    cm[:_lut] = colors

    # prepare images
    A_orig = wplotim(yt)
    A_orig_pruned = wplotim(yt_pruned)
    A = sign(A_orig) .* log2(1 + abs(A_orig))
    A_pruned = sign(A_orig_pruned) .* log2(1 + abs(A_orig_pruned))
    r = max(maximum(abs(A)), maximum(abs(A_pruned)))
    A_mask = 1 - abs(sign(A))
    A_pruned_mask = 1 - abs(sign(A_pruned))

    f, ax = subplots(3, 1, sharex=true)

    ax[1][:scatter](x, y_pruned, c="black", s=(error * 10000000))
    ax[1][:plot](x, y, c="blue")
    ax[1][:plot](x, y_pruned, c="red")
    ax[1][:set_yscale]("log")
    ax[1][:set_xlim]([year_first, year_first + n - 1])
    ax[1][:set_xlabel]("year")
    ax[1][:set_ylabel](L"signal $x$")
    ax[1][:set_title]("input/output function")

    img = ax[2][:imshow](A, aspect="auto", interpolation="none", extent=[year_first, year_first + n - 1, J - 0.5, -0.5], vmin=-r, vmax=r, cmap="Spectral")
    img_mask = ax[2][:imshow](A_mask, aspect="auto", interpolation="none", extent=[year_first, year_first + n - 1, J - 0.5, -0.5], cmap=cm)
    ax[2][:set_xlabel]("year")
    ax[2][:set_ylabel](L"level $j$")
    ax[2][:set_title](@sprintf("wavelet (size=%i)", count(e -> abs(e) > 0, yt)))

    img_pruned = ax[3][:imshow](A_pruned, aspect="auto", interpolation="none", extent=[year_first, year_first + n - 1, J - 0.5, -0.5], vmin=-r, vmax=r, cmap="Spectral")
    img_pruned_mask = ax[3][:imshow](A_pruned_mask, aspect="auto", interpolation="none", extent=[year_first, year_first + n - 1, J - 0.5, -0.5], cmap=cm)
    ax[3][:set_xlabel]("year")
    ax[3][:set_ylabel](L"level $j$")
    ax[3][:set_title](@sprintf("wavelet (size=%i)", count(e -> abs(e) > 0, yt_pruned)))

    f[:set_size_inches](12, 12)
    f[:subplots_adjust](left=0.05, right=0.9, top=0.9, bottom=0.05, hspace=0.5)
    cbar_ax = f[:add_axes]([0.95, 0.15, 0.01, 0.7])
    f[:colorbar](img, cax=cbar_ax)

    title = @sprintf("%s (threshold=%f, error=%f)", token, threshold, sum(error))
    f[:suptitle](title, fontsize=24)
end

function plot_wavelet_report(data, map_s2i, year_first, ngram, threshold)
    # get data
    y = convert(Array{Float64}, data[:,map_s2i[ngram]])

    # run
    y_pruned, error, yt, yt_pruned = do_wavelet_pruning(y, threshold)

    # plot
    plot_wavelet_report_helper(y, y_pruned, yt, yt_pruned, error, year_first, ngram, threshold)
end

function do_wavelet_pruning(y, threshold)
    # transform
    y_log = log10(1.0 .+ y)
    scale = sum(y_log)
    wt = wavelet(WT.haar)
    yt = dwt(y_log, wt)

    # pruning
    yt_pruned = copy(yt)
    prune_threshold!(yt_pruned, threshold * scale)

    # transform back
    y_pruned_log = idwt(yt_pruned, wt)
    y_pruned = 10 .^ (y_pruned_log) .- 1.0

    # error
    error = ((y_log - y_pruned_log) ./ scale) .^ 2

    return y_pruned, error, yt, yt_pruned
end

function prune_threshold!(yt, threshold)
    depth = floor(log2(length(yt)))
    start_i = 1
    start_level = 1
    return prune_threshold!(yt, depth, start_i, start_level, threshold)
end

function prune_threshold!(yt, depth, i, level, threshold)
    if i + 1 <= length(yt)
        level_next = level + 1
        influence = 2^(depth - level + 1)

        (l_pruned, l_error) = prune_threshold!(yt, depth, i * 2, level_next, threshold)
        (r_pruned, r_error) = prune_threshold!(yt, depth, i * 2 + 1, level_next, threshold)

        sub_pruned = l_pruned && r_pruned
        sub_error = l_error + r_error

        this_error = sqrt(influence) * abs(yt[i + 1]) + sub_error

        if sub_pruned && (this_error < threshold)
            yt[i + 1] = 0
            return (true, this_error)
        else
            return (false, sub_error)
        end
    else
        return (true, 0.0)
    end
end
