for f in $(ls $1)
do
    echo "$(basename $f .log),$(grep -B1 'virt.' "$1/$f" | grep 'occ.' | tail -1 | rev | cut -b 1-9 | rev )"
done
