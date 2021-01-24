# INPUT $1 = gaussian log input file

N_ATOMS=$(grep "NAtoms=" $1 | tr -s ' ' | cut -d ' ' -f 3)

{ echo "atom_number,atom_symbol,x,y,z" & grep --after-context=$N_ATOMS "Multiplicity" $1 | tail -n +2 | cat -n | awk '{$1=$1};1' | tr -s '[:blank:]' ','; } | cat