# Inputs: 
# $1 = inputfolder 
# $2 = outputfolder

PARSER="D:/Projects/Y4Project/python/parse_orbitals.py"

for file in $1
do
    "C:/Python38/python.exe" $PARSER $file > $2/$(basename -- $file .log).json
done