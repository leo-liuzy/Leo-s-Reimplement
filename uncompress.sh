name=outputorder1_lr3e-5_bs32_enc-dec

tar -xvzf "$name"_results.tgz

for seedidx in {1..3}
do 
    for shuffleidx in {1..3}
    do 
        dirname="$name"_seed"$seedidx"_shuffle"$shuffleidx"
        mkdir $dirname
        tarname="$dirname"_results.tgz
        mv $tarname $dirname/
    done
done    

