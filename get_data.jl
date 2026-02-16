using Downloads
using JLD2
using ZipArchives: ZipReader, zip_names, zip_readentry

data = take!(Downloads.download("https://surfdrive.surf.nl/public.php/dav/files/LiqmgaK436rFygR/?accept=zip", IOBuffer()))
archive = ZipReader(data)
# zip_names(archive)
iobuf = zip_readentry(archive, "data/p13_t0.10_nu5e-04_i508.jld2") |> IOBuffer

jldopen(iobuf) do f
    f["u"]
end