for f in $(ls $1)
do
    echo "$(basename $f .log),$(grep -B1 'virt.' "$1/$f" | grep 'occ.' | tail -1 | rev | cut -b 1-9 | rev )"
done

#ALANINE="NC(C(=O)O)C"
ALANINE="N[CH](C)C(=O)O"
MOL2="CCC(=O)Cl"
# -xf FORMAT     for all formats use babel -L fingerprints
# obabel -:$ALANINE -:$MOL2 -ofpt -xfFP3
# obabel -:$ALANINE -O "alanine.svg"